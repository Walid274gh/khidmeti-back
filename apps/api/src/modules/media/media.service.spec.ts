// Check for parseCloudinaryUrl — the trust boundary behind the portfolio
// endpoint. It decides whether a client-supplied URL is our own media (and
// whose), so a hole here means arbitrary images on a public worker profile.

import { MediaService, parseCloudinaryUrl } from './media.service';
import { CloudinaryConfigService } from '../../config/cloudinary.config';

const URL_OK =
  'https://res.cloudinary.com/df9mahgkj/image/upload/v1719312345/service-media/uid-42/1719312345678_a1b2.jpg';

describe('parseCloudinaryUrl', () => {
  it('splits cloud name, resource type and extension-less public id', () => {
    expect(parseCloudinaryUrl(URL_OK)).toEqual({
      cloudName:    'df9mahgkj',
      resourceType: 'image',
      publicId:     'service-media/uid-42/1719312345678_a1b2',
    });
  });

  it('keeps the uploader id in the public id (ownership proof)', () => {
    expect(parseCloudinaryUrl(URL_OK)!.publicId).toContain('/uid-42/');
  });

  it('rejects foreign hosts, http, and non-Cloudinary shapes', () => {
    expect(parseCloudinaryUrl('https://evil.example.com/a.jpg')).toBeNull();
    expect(parseCloudinaryUrl(URL_OK.replace('https', 'http'))).toBeNull();
    expect(parseCloudinaryUrl('https://res.cloudinary.com/c/image/fetch/v1/x.jpg')).toBeNull();
    expect(parseCloudinaryUrl('')).toBeNull();
  });

  it('rejects a host that merely ends with the real one', () => {
    expect(
      parseCloudinaryUrl('https://res.cloudinary.com.evil.test/c/image/upload/v1/x.jpg'),
    ).toBeNull();
  });

  it('strips the delivery extension for video too', () => {
    const v = parseCloudinaryUrl(URL_OK.replace('/image/', '/video/').replace('.jpg', '.mp4'));
    expect(v!.resourceType).toBe('video');
    expect(v!.publicId.endsWith('_a1b2')).toBe(true);
  });

  it('keeps the extension for raw — upload() bakes it into the public id', () => {
    const r = parseCloudinaryUrl(
      URL_OK.replace('/image/', '/raw/').replace('.jpg', '.pdf'),
    );
    expect(r!.resourceType).toBe('raw');
    expect(r!.publicId.endsWith('.pdf')).toBe(true);
  });
});

// The gate that decides whether a client-supplied URL may be published on a
// public worker profile. Everything it rejects is something a worker could
// otherwise put in front of every client.
describe('isOwnPortfolioImage', () => {
  const config = {
    cloudName:       'df9mahgkj',
    folderPortfolio: 'portfolio',
  } as CloudinaryConfigService;
  const media = new MediaService(config);
  const mine =
    'https://res.cloudinary.com/df9mahgkj/image/upload/v1/portfolio/uid-42/1719_a1b2.jpg';

  it('accepts this worker own portfolio image', () => {
    expect(media.isOwnPortfolioImage(mine, 'uid-42')).toBe(true);
  });

  it('rejects another worker photo', () => {
    expect(media.isOwnPortfolioImage(mine, 'uid-99')).toBe(false);
  });

  it('rejects a foreign cloud account', () => {
    expect(
      media.isOwnPortfolioImage(mine.replace('df9mahgkj', 'attacker'), 'uid-42'),
    ).toBe(false);
  });

  it('rejects other folders — an avatar or an ID scan is not gallery content', () => {
    expect(
      media.isOwnPortfolioImage(mine.replace('/portfolio/', '/profile-images/'), 'uid-42'),
    ).toBe(false);
    expect(
      media.isOwnPortfolioImage(mine.replace('/portfolio/', '/verification-docs/'), 'uid-42'),
    ).toBe(false);
  });

  it('rejects a uid that is only a prefix of the real folder segment', () => {
    expect(media.isOwnPortfolioImage(mine, 'uid-4')).toBe(false);
  });

  it('rejects non-image resource types and junk', () => {
    expect(media.isOwnPortfolioImage(mine.replace('/image/', '/video/'), 'uid-42')).toBe(false);
    expect(media.isOwnPortfolioImage('not-a-url', 'uid-42')).toBe(false);
  });
});
