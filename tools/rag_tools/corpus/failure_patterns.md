# Known failure patterns (synthetic)

## [failure_pattern] Target service not running
tags: connection refused, transport error, environment
Symptoms: ConnectionError for every action. The target process is down or the base URL/port is wrong. Check the health endpoint and QA_API_BASE_URL.

## [failure_pattern] Slow dependency causes timeouts
tags: timeout, read timeout, latency, network
Symptoms: ReadTimeout or ConnectTimeout on some actions only. Often a slow downstream dependency. Re-run with a higher timeout and inspect dependency latency.

## [failure_pattern] Missing or expired credentials
tags: 401, 403, unauthorized, forbidden, token, configuration
Symptoms: HTTP 401/403 where success was expected. Test credentials are missing, expired or lack a role.

## [failure_pattern] Validation not enforced on server
tags: 201, 200, validation, limit, boundary, application defect
Symptoms: a request that should be rejected (expected 4xx) succeeds with 2xx. A server-side rule such as a quantity limit is missing.

## [failure_pattern] Exhausted or polluted test data
tags: 409, 404, out of stock, not found, test data
Symptoms: 409 conflict or 404 for data the scenario assumed to exist. Earlier runs consumed stock or deleted records. Re-seed test data.

## [failure_pattern] Internal server error
tags: 500, server error, exception, application defect
Symptoms: HTTP 500 from the application. An unhandled exception; collect server logs around the request time.

## [failure_pattern] Upstream gateway errors
tags: 502, 503, 504, gateway, dependency
Symptoms: HTTP 502/503/504. A dependency or gateway is unavailable. Check upstream status before treating as a product defect.

## [failure_pattern] Selector or locator drift
tags: selector, locator, timeout, browser, automation defect
Symptoms: a browser step times out waiting for an element although the page loaded. Update the locator to a stable test id.

## [failure_pattern] Contract drift in response body
tags: json path, missing field, schema, contract, automation
Symptoms: an expected JSON field is missing from a successful response. Compare the response with the published OpenAPI schema.

## [failure_pattern] Redirect to unexpected location
tags: redirect, 301, 302, configuration
Symptoms: redirect responses where a direct response was expected. Often a wrong base URL or an auth redirect.