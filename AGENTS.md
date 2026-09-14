# Repository Instructions & Quality Contract

## Verification Protocol for Code Changes
Whenever you make ANY code change, bug fix, or refactor in this repository:
1. **Self-Verification**: You MUST run `./check.sh` yourself using `run_command`.
2. **Auto-Fix**: If `./check.sh` reports any failure (syntax error, lint error, test failure, clean CI isolation issue, or build failure), you MUST investigate and fix the errors yourself.
3. **No Assumptions**: Do not ask or tell the user to push until `./check.sh` has executed in your tools and returned exit code 0 (`ALL CHECKS PASSED`).
4. **Final Confirmation**: Only after `./check.sh` passes 100% may you confirm to the user that the code is clean, safe, and ready to push.
