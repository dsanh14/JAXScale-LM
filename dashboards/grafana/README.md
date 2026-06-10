# Grafana

A Grafana dashboard is an optional extension (see docs/limitations.md §
roadmap). The Prometheus scrape config in `../prometheus.yml` already
exposes all `jaxscale_*` series; point any Grafana instance at the
Prometheus container (`http://prometheus:9090`) to chart them.

Useful starting queries:

- `rate(jaxscale_generated_tokens_total[1m])` — token throughput
- `histogram_quantile(0.95, rate(jaxscale_request_latency_seconds_bucket[5m]))` — p95 latency
- `jaxscale_active_requests` — concurrency
