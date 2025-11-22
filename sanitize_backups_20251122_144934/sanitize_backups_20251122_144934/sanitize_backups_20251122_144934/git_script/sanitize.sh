#!/bin/bash

# ============================================================
# ASNet Anonymization Script
# Safely removes personal identifiers from text files.
# ============================================================

TERMS=("xxxxxx" "xxxxxxkumar" "skumar4")
EXTENSIONS=("py" "yaml" "yml" "sh" "txt")
BACKUP_DIR="sanitize_backups_$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo " 🔍  SCANNING FOR PERSONAL IDENTIFIERS"
echo "============================================================"
echo

FOUND=0
for term in "${TERMS[@]}"; do
    echo "Searching for: \"$term\""
    matches=$(grep -Rni "$term" . --exclude-dir={.git,__pycache__} || true)
    if [[ ! -z "$matches" ]]; then
        FOUND=1
        echo "$matches"
    else
        echo "  → No matches found."
    fi
    echo
done

if [[ $FOUND -eq 0 ]]; then
    echo "✨ No occurrences found. No sanitization needed."
    exit 0
fi

echo "============================================================"
echo " ⚠️  Matches found above."
echo "     Files WILL BE MODIFIED if you continue."
echo "============================================================"
read -p "Proceed with anonymization? (y/n): " confirm

if [[ "$confirm" != "y" ]]; then
    echo "Aborted."
    exit 0
fi

echo
echo "============================================================"
echo " 🛡  Creating backup directory: $BACKUP_DIR"
echo "============================================================"
mkdir -p "$BACKUP_DIR"

echo
echo "============================================================"
echo " ✏️  Sanitizing files..."
echo "============================================================"

for term in "${TERMS[@]}"; do
    for ext in "${EXTENSIONS[@]}"; do
        
        # find all matching files of allowed extensions
        files=$(grep -Rl "$term" . --include="*.$ext" --exclude-dir={.git,__pycache__} || true)

        for file in $files; do
            # Create backup
            backup_file="$BACKUP_DIR/$(echo "$file" | sed 's|^\./||')"
            mkdir -p "$(dirname "$backup_file")"
            cp "$file" "$backup_file"

            # Replace term with xxxxxx
            sed -i "s/$term/xxxxxx/g" "$file"
            echo "Sanitized: $file (removed \"$term\")"
        done
    done
done

echo
echo "============================================================"
echo " 🎉  DONE!"
echo " Backups stored in: $BACKUP_DIR"
echo "============================================================"


# ======================================== Make script executable ========================================
# chmod +x sanitize.sh

# ./sanitize.sh


