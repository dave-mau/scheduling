docker exec ci-container pytest

success=$?
if [[ $success -eq 0 ]]; then
    echo "Unit Tests Passed"
    exit 0
else
    echo "Unit Tests Failed"
    exit 1
fi