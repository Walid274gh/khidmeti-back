// apps/api/src/dto/update-portfolio.dto.ts
//
// Body of PUT /workers/:id/portfolio — the worker's full, ordered list of
// showcase photos. The client uploads the bytes through /media/upload/image
// first and sends back the returned URLs.
//
// ArrayMaxSize here is only a payload guard; the real cap is the pack's
// portfolioQuota, enforced in UsersService.setPortfolio.

import { IsArray, IsUrl, ArrayMaxSize } from 'class-validator';

export class UpdatePortfolioDto {
  @IsArray()
  @ArrayMaxSize(50)
  @IsUrl({}, { each: true })
  portfolio: string[];
}
