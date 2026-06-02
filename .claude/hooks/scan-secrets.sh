#!/bin/bash
# Pre-write secret scanner
# Receives Write tool input as JSON on stdin
# Exit 2 = block the write | Exit 0 = allow

INPUT=$(cat)

# Extract file path and content from the JSON tool input
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    print(json.load(sys.stdin).get('file_path', ''))
except: pass
" 2>/dev/null)

CONTENT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    print(json.load(sys.stdin).get('content', ''))
except: pass
" 2>/dev/null)

# Skip if no content
if [ -z "$CONTENT" ]; then exit 0; fi

# Never scan the scan-secrets script itself
if echo "$FILE_PATH" | grep -q "scan-secrets"; then exit 0; fi

BLOCKED=0

check_pattern() {
    local PATTERN="$1"
    local LABEL="$2"
    if echo "$CONTENT" | grep -qiE "$PATTERN" 2>/dev/null; then
        echo "SECURITY BLOCK: $LABEL detected in $FILE_PATH"
        echo "Use environment variables instead (the repo must run with no API key through Phase 6)."
        BLOCKED=1
    fi
}

# AWS credentials
check_pattern 'AKIA[0-9A-Z]{16}' "AWS Access Key ID"
check_pattern 'aws_secret_access_key\s*[=:]\s*[A-Za-z0-9+/]{40}' "AWS Secret Access Key"

# Generic high-confidence patterns (LLM / API-embedding phases add keys)
check_pattern 'password\s*[=:]\s*["\x27][^"\x27]{8,}["\x27]' "Hardcoded password"
check_pattern '(api_key|apikey|api-key)\s*[=:]\s*["\x27][A-Za-z0-9_\-]{16,}' "API key"
check_pattern 'sk-[A-Za-z0-9]{20,}' "OpenAI-style API key"
check_pattern 'sk-ant-[A-Za-z0-9_\-]{20,}' "Anthropic API key"
check_pattern 'Bearer [A-Za-z0-9_\-\.]{20,}' "Bearer token"

if [ $BLOCKED -eq 1 ]; then
    echo ""
    echo "Write blocked. Fix the secret(s) above before writing."
    exit 2
fi

exit 0
