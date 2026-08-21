import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { GeographicCell, GeographicCellDocument } from '../../schemas/geographic-cell.schema';
import {
  User,
  UserDocument,
  UserRole,
  subscriptionVisibilityFilter,
  PORTFOLIO_PREVIEW_COUNT,
} from '../../schemas/user.schema';

export interface AssignCellResult {
  cellId: string;
  wilayaCode: number;
  geoHash: string;
}

/**
 * LocationService handles geographic cell assignment for workers.
 * All queries against workers use the unified 'users' collection with
 * role='worker' filter.
 */
@Injectable()
export class LocationService {
  private readonly logger = new Logger(LocationService.name);

  private static readonly CELL_PRECISION    = 2;
  private static readonly DEFAULT_RADIUS_KM = 5.0;

  constructor(
    @InjectModel(GeographicCell.name)
    private readonly cellModel: Model<GeographicCellDocument>,
    @InjectModel(User.name)
    private readonly userModel: Model<UserDocument>,
  ) {}

  async assignWorkerToCell(
    workerId: string,
    latitude: number,
    longitude: number,
    wilayaCode: number,
  ): Promise<AssignCellResult> {
    try {
      const cellId  = this.buildCellId(latitude, longitude, wilayaCode);
      const geoHash = this.encodeGeoHash(latitude, longitude, 6);

      await this.ensureCellExists(cellId, latitude, longitude, wilayaCode);

      await this.userModel
        .updateOne(
          { _id: workerId, role: UserRole.Worker },
          { cellId, wilayaCode, geoHash, lastCellUpdate: new Date() },
        )
        .exec();

      return { cellId, wilayaCode, geoHash };
    } catch (err) {
      this.logger.error(`LocationService.assignWorkerToCell(${workerId}) failed`, err);
      throw err;
    }
  }

  async getWorkersInCell(
    cellId: string,
    serviceType?: string,
    onlineOnly = false,
    limit = 50,
    viewerId?: string,
  ): Promise<UserDocument[]> {
    try {
      const query: Partial<Record<string, unknown>> = {
        cellId,
        role: UserRole.Worker,
      };
      if (serviceType) query['profession'] = serviceType;
      if (onlineOnly)  query['isOnline']   = true;
      // Visibility gate — same contract as search (subscriptionVisibilityFilter):
      // active + not expired + pack allowed today + daily quota not exhausted.
      Object.assign(query, subscriptionVisibilityFilter(new Date()));
      // Business-account view: server-enforced from the viewer's persisted role,
      // so a client can never spoof its way past — and Business always sees the
      // Expert-only subset. lean+projection skips PII decryption.
      if (viewerId && (await this.isBusinessViewer(viewerId))) {
        query['b2bAccess'] = true;
      }
      // Business/Expert priority: they fill the limited result window first.
      // portfolio is $slice'd: the map sheet previews 3 thumbnails, so shipping
      // up to 26 URLs × 50 workers per pan would be ~150 kB of dead payload on
      // a mobile connection. The full gallery comes with GET /workers/:id.
      // ($slice alone is not an inclusive projection — every other field still
      // comes back.)
      return this.userModel
        .find(query)
        .slice('portfolio', PORTFOLIO_PREVIEW_COUNT)
        .sort({ searchPriority: -1 })
        .limit(limit)
        .exec();
    } catch (err) {
      this.logger.error(`LocationService.getWorkersInCell(${cellId}) failed`, err);
      throw err;
    }
  }

  /** True when the authenticated viewer is a Business (B2B) account. */
  private async isBusinessViewer(viewerId: string): Promise<boolean> {
    const doc = await this.userModel.findById(viewerId).select('role').lean().exec();
    return (doc as { role?: string } | null)?.role === UserRole.Business;
  }

  getAdjacentCellIds(cellId: string): string[] {
    const parts = cellId.split('_');
    if (parts.length !== 3) return [];

    const [wilayaCode, latStr, lngStr] = parts;
    const lat  = parseFloat(latStr);
    const lng  = parseFloat(lngStr);
    const step = Math.pow(10, -LocationService.CELL_PRECISION);
    const prec = LocationService.CELL_PRECISION;

    const ids: string[] = [];
    for (let dLat = -1; dLat <= 1; dLat++) {
      for (let dLng = -1; dLng <= 1; dLng++) {
        if (dLat === 0 && dLng === 0) continue;
        const adjLat = +(lat + dLat * step).toFixed(prec);
        const adjLng = +(lng + dLng * step).toFixed(prec);
        ids.push(`${wilayaCode}_${adjLat.toFixed(prec)}_${adjLng.toFixed(prec)}`);
      }
    }
    return ids;
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  private buildCellId(lat: number, lng: number, wilayaCode: number): string {
    const p    = LocationService.CELL_PRECISION;
    const rLat = +lat.toFixed(p);
    const rLng = +lng.toFixed(p);
    return `${wilayaCode}_${rLat.toFixed(p)}_${rLng.toFixed(p)}`;
  }

  private async ensureCellExists(
    cellId: string,
    lat: number,
    lng: number,
    wilayaCode: number,
  ): Promise<void> {
    const exists = await this.cellModel.exists({ _id: cellId });
    if (!exists) {
      const adjacentCellIds = this.getAdjacentCellIds(cellId);
      await this.cellModel
        .findByIdAndUpdate(
          cellId,
          {
            wilayaCode,
            centerLat: +lat.toFixed(LocationService.CELL_PRECISION),
            centerLng: +lng.toFixed(LocationService.CELL_PRECISION),
            radius:    LocationService.DEFAULT_RADIUS_KM,
            adjacentCellIds,
          },
          { upsert: true },
        )
        .exec();
    }
  }

  private encodeGeoHash(lat: number, lng: number, precision: number): string {
    const BASE32  = '0123456789bcdefghjkmnpqrstuvwxyz';
    let   hash    = '';
    let   isEven  = true;
    let   bit     = 0;
    let   ch      = 0;
    let   latMin  = -90.0, latMax = 90.0;
    let   lngMin  = -180.0, lngMax = 180.0;

    while (hash.length < precision) {
      let mid: number;
      if (isEven) {
        mid = (lngMin + lngMax) / 2;
        if (lng >= mid) { ch |= (1 << (4 - bit)); lngMin = mid; } else { lngMax = mid; }
      } else {
        mid = (latMin + latMax) / 2;
        if (lat >= mid) { ch |= (1 << (4 - bit)); latMin = mid; } else { latMax = mid; }
      }
      isEven = !isEven;
      if (bit < 4) { bit++; } else { hash += BASE32[ch]; bit = 0; ch = 0; }
    }
    return hash;
  }
}
