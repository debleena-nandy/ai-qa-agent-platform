# QA checklists (synthetic, written for this portfolio)

## [checklist] E-commerce order placement
tags: order, cart, checkout, buy, purchase, stock, payment
- inventory reserved atomically with order creation
- idempotent order creation for retried submissions
- server-side price calculation ignores client prices
- declined and pending payment outcomes
- quantity limits per order line
- order ownership and authorization on read

## [checklist] Order cancellation and refunds
tags: cancel, refund, return, order
- only cancellable states can be cancelled
- stock restored after cancellation
- refund issued exactly once
- cancellation of another customer's order is rejected

## [checklist] Authentication and sessions
tags: login, sign in, password, session, token, credentials
- generic error message for wrong credentials
- account lockout after repeated failures
- session expiry and logout invalidation
- tokens never logged or returned in URLs
- brute-force rate limiting

## [checklist] Password reset
tags: password, reset, forgot, email, link
- reset link single-use and time-limited
- no account enumeration via reset form
- old sessions invalidated after reset

## [checklist] Profile and account settings
tags: profile, account, settings, avatar, photo, preferences
- required field and format validation
- only the owner can edit the profile
- upload type and size limits
- stored values escaped on display

## [checklist] Payments
tags: payment, card, upi, pay, transaction, refund, webhook
- duplicate transaction detection
- timeout and retry without double charge
- webhook signature validation
- currency and amount precision

## [checklist] Reporting and exports
tags: report, export, csv, download, monthly
- large dataset export does not time out
- export respects user permissions
- date range boundaries inclusive and correct
- formula injection neutralised in CSV cells

## [checklist] Search and filtering
tags: search, filter, sort, query, list
- empty and no-result queries
- pagination boundaries
- special characters and injection in queries