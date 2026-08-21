// ══════════════════════════════════════════════════════════════════════════════
// AdminController — /admin/*
//
// Every route is protected by FirebaseAuthGuard (valid ID token) + AdminGuard
// (caller's Mongo role must be 'admin' and not banned). Applied at class level
// so no route can accidentally be left open.
// ══════════════════════════════════════════════════════════════════════════════

import {
  Body,
  Controller,
  Delete,
  Get,
  HttpCode,
  HttpStatus,
  Param,
  Patch,
  Post,
  Query,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';
import { FirebaseAuthGuard } from '../../common/guards/firebase-auth.guard';
import { AdminGuard } from '../../common/guards/admin.guard';
import { AdminService } from './admin.service';
import {
  BroadcastDto,
  ListBidsQueryDto,
  ListRequestsQueryDto,
  ListUsersQueryDto,
  ListWorkersQueryDto,
  SetBanDto,
  SetOnlineDto,
  SetRoleDto,
  SetVerificationDto,
  SetVerifiedDto,
  UpdateUserAdminDto,
} from './dto/admin-query.dto';
import {
  CreateProfessionDto,
  UpdateProfessionDto,
} from './dto/profession-admin.dto';

@ApiTags('admin')
@ApiBearerAuth()
@Controller('admin')
@UseGuards(FirebaseAuthGuard, AdminGuard)
export class AdminController {
  constructor(private readonly admin: AdminService) {}

  // ── Dashboard ──────────────────────────────────────────────────────────────
  @Get('stats')
  getStats() {
    return this.admin.getStats();
  }

  // ── Users ──────────────────────────────────────────────────────────────────
  @Get('users')
  listUsers(@Query() q: ListUsersQueryDto) {
    return this.admin.listUsers(q);
  }

  @Get('users/:id')
  getUser(@Param('id') id: string) {
    return this.admin.getUser(id);
  }

  @Patch('users/:id')
  updateUser(@Param('id') id: string, @Body() dto: UpdateUserAdminDto) {
    return this.admin.updateUser(id, dto);
  }

  @Patch('users/:id/role')
  setRole(@Param('id') id: string, @Body() dto: SetRoleDto) {
    return this.admin.setRole(id, dto.role);
  }

  @Patch('users/:id/ban')
  setBan(@Param('id') id: string, @Body() dto: SetBanDto) {
    return this.admin.setBan(id, dto.isBanned);
  }

  /** Approve or reject a submitted verification-document set. */
  @Patch('users/:id/verification')
  setVerification(@Param('id') id: string, @Body() dto: SetVerificationDto) {
    return this.admin.setVerification(id, dto.status, dto.note);
  }

  @Delete('users/:id')
  @HttpCode(HttpStatus.OK)
  deleteUser(@Param('id') id: string) {
    return this.admin.deleteUser(id);
  }

  // ── Workers ────────────────────────────────────────────────────────────────
  @Get('workers')
  listWorkers(@Query() q: ListWorkersQueryDto) {
    return this.admin.listWorkers(q);
  }

  @Patch('workers/:id/verify')
  setVerified(@Param('id') id: string, @Body() dto: SetVerifiedDto) {
    return this.admin.setVerified(id, dto.isVerified);
  }

  @Patch('workers/:id/status')
  setOnline(@Param('id') id: string, @Body() dto: SetOnlineDto) {
    return this.admin.setOnline(id, dto.isOnline);
  }

  // ── Service requests ─────────────────────────────────────────────────────────
  @Get('service-requests')
  listRequests(@Query() q: ListRequestsQueryDto) {
    return this.admin.listRequests(q);
  }

  @Get('service-requests/:id')
  getRequest(@Param('id') id: string) {
    return this.admin.getRequest(id);
  }

  @Post('service-requests/:id/cancel')
  @HttpCode(HttpStatus.OK)
  cancelRequest(@Param('id') id: string) {
    return this.admin.cancelRequest(id);
  }

  // ── Bids ─────────────────────────────────────────────────────────────────────
  @Get('bids')
  listBids(@Query() q: ListBidsQueryDto) {
    return this.admin.listBids(q);
  }

  // ── Professions (CRUD) ────────────────────────────────────────────────────────
  @Get('professions')
  listProfessions() {
    return this.admin.listProfessions();
  }

  @Post('professions')
  createProfession(@Body() dto: CreateProfessionDto) {
    return this.admin.createProfession(dto);
  }

  @Patch('professions/:key')
  updateProfession(@Param('key') key: string, @Body() dto: UpdateProfessionDto) {
    return this.admin.updateProfession(key, dto);
  }

  @Delete('professions/:key')
  @HttpCode(HttpStatus.OK)
  deleteProfession(@Param('key') key: string) {
    return this.admin.deleteProfession(key);
  }

  // ── Broadcast ────────────────────────────────────────────────────────────────
  @Post('broadcast')
  @HttpCode(HttpStatus.OK)
  broadcast(@Body() dto: BroadcastDto) {
    return this.admin.broadcast(dto);
  }
}
