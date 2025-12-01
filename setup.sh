#/bin/bash
GIT_FOLDER="$(pwd)"
PATH_TO_HTML="$(dirname "$GIT_FOLDER")/scToolkit/scToolkit_doc.html"
ln -sf "$GIT_FOLDER/docs/_build/html/index.html" "$PATH_TO_HTML"
