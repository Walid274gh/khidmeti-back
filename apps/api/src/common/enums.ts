export enum ServiceStatus {
  Open = 'open',
  AwaitingSelection = 'awaitingSelection',
  BidSelected = 'bidSelected',
  InProgress = 'inProgress',
  Completed = 'completed',
  Cancelled = 'cancelled',
  Expired = 'expired',
}

export enum ServicePriority {
  Low = 'low',
  Normal = 'normal',
  High = 'high',
  Urgent = 'urgent',
}

export enum BidStatus {
  Pending = 'pending',
  Accepted = 'accepted',
  Declined = 'declined',
  Withdrawn = 'withdrawn',
  Expired = 'expired',
}

export enum MediaType {
  Image = 'image',
  Video = 'video',
  Text = 'text',
}
