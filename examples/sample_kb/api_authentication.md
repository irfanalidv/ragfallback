# API Authentication

All API requests must include a valid Bearer token in the Authorization header. Tokens are issued per account and expire after 90 days.

To generate a new API token, navigate to Settings → API Keys and click "Generate Token". Each account may have up to 5 active tokens simultaneously.

Tokens must be kept confidential. Do not share tokens in public repositories or client-side code. Compromised tokens should be revoked immediately via the dashboard.

Rate limiting is enforced per token. Exceeding the rate limit returns HTTP 429 with a Retry-After header indicating when the next request may be made.

OAuth 2.0 is supported for third-party integrations. Use the authorization code flow for server-side apps and the PKCE flow for single-page applications.
