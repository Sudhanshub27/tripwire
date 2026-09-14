# Pre-Push & Code Change Verification Rule

Whenever any code changes, fixes, or refactors are made to this repository:
1. You MUST execute `./check.sh` before completing your turn or asking the user to push.
2. If `./check.sh` encounters any errors, you MUST analyze and fix the errors immediately.
3. Keep iterating until `./check.sh` passes 100% with exit code 0.
4. Only when `./check.sh` reports `ALL CHECKS PASSED` should you tell the user that the code is ready to push.
