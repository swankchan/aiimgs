git filter-branch --force --index-filter "git rm -r --cached --ignore-unmatch 'images backup'" --prune-empty --tag-name-filter cat -- --all
