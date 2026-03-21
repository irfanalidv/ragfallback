# Rate Limits

Free accounts are limited to 100 API requests per minute and 10 000 per day. Pro accounts have a limit of 1 000 requests per minute and 500 000 per day. Enterprise accounts have custom limits negotiated at contract time.

When a rate limit is exceeded, the API returns HTTP status 429. The response includes a Retry-After header with the number of seconds to wait before retrying. Clients should implement exponential backoff to avoid further throttling.

Bulk operations (batch endpoints) count as a single request regardless of the number of items in the batch, up to a maximum batch size of 500 items. Batches exceeding 500 items must be split into multiple requests.

Streaming endpoints count each chunk as a separate request against rate limits. Monitor usage via the X-RateLimit-Remaining response header.
