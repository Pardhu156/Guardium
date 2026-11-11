# Guardium Gmail defensive verification reports

For each completed benign validation, record the run ID, test type, execution
mode, model identifier (if used), synthetic user request (if used), Gmail
message ID, tool names, guard decisions, whether a send was proposed or
executed, task outcome, latency, and sanitized errors. Use null for fields
that were not measured. Store local records under the ignored `logs/` directory.
Exclude Gmail addresses, message bodies, OAuth details, and model traces.

The standalone Gmail check reports a matching message ID. The mock suite
reports its test outcomes. Neither produces measurements of attack success.
