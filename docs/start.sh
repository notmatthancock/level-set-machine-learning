python -c "import sys; sys.exit(sys.version_info.major)"
version=$?
/opt/google/chrome/chrome http://localhost:8000
if [ "$version" = 2 ]; then
    python -m SimpleHTTPServer
else
    python -c "print ..."
fi
