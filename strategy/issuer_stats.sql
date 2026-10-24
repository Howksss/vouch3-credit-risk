-- Issuer track record for rules_v2. Column aliases are the keys compute_grade_v2
-- reads from issuer_stats. Runs as is on PostgreSQL and on SQLite >= 3.30.
SELECT
    count(id)                                               AS total_issues,
    count(id) FILTER (WHERE status = 'active')              AS active_issues,
    count(id) FILTER (WHERE status = 'redeemed')            AS redeemed_issues,
    count(id) FILTER (WHERE status = 'defaulted')           AS defaulted_issues,
    count(id) FILTER (WHERE status = 'cancelled')           AS cancelled_issues,
    avg(coupon_rate) FILTER (WHERE coupon_rate IS NOT NULL) AS avg_coupon
FROM instrument
WHERE issuer_id = :issuer_id
