# vhs script for the README streaming GIF:  vhs docs/assets/demo.tape
# Prereqs: gateway on :8080, a worker, and any engine (make mock works).
Output docs/assets/streaming-demo.gif

Set FontSize 16
Set Width 1200
Set Height 640
Set Theme "Catppuccin Mocha"
Set Padding 18
Set TypingSpeed 40ms

Type "# one OCR model, four serving stacks — streaming a PDF page by page"
Enter
Sleep 1s

Type "curl -s -X POST localhost:8080/v1/ocr -H 'X-API-Key: dev-key' -F file=@sample.pdf | jq"
Enter
Sleep 2s

Type "curl -N localhost:8080/v1/ocr/$JOB/stream -H 'X-API-Key: dev-key'"
Enter
Sleep 12s

Type "curl -s localhost:8080/v1/ocr/$JOB -H 'X-API-Key: dev-key' | jq '.page_count, .timings'"
Enter
Sleep 3s
