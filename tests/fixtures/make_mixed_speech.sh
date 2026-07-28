#!/usr/bin/env bash
# Builds a mixed English/Hebrew clip for the integration test, using the macOS
# speech synthesiser. Hebrew needs the "Carmit" voice (System Settings →
# Accessibility → Spoken Content → System Voice → Manage Voices).
#
#   ./tests/fixtures/make_mixed_speech.sh
#
# Produces tests/fixtures/mixed_en_he.wav: English, then Hebrew, then English.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"

say -v Samantha -o /tmp/_en1.aiff \
  "Good morning everyone, let's start the standup. I finished the recorder work yesterday and it is now capturing both sides of the call."
say -v Carmit -o /tmp/_he1.aiff \
  "בוקר טוב, אני רוצה לדבר על הפרויקט החדש. סיימתי את העבודה על ההקלטה והכל עובד טוב מאוד עכשיו."
say -v Samantha -o /tmp/_en2.aiff \
  "That sounds good. I will review the pull request this afternoon and we can ship it tomorrow."

for n in _en1 _he1 _en2; do
  ffmpeg -y -v error -i /tmp/$n.aiff -ac 1 -ar 16000 /tmp/$n.wav
done
: > /tmp/_list.txt
for i in 1 2; do printf "file '%s'\n" /tmp/_en1.wav /tmp/_he1.wav /tmp/_en2.wav >> /tmp/_list.txt; done
ffmpeg -y -v error -f concat -safe 0 -i /tmp/_list.txt -ac 1 -ar 16000 mixed_en_he.wav
rm -f /tmp/_en1.* /tmp/_he1.* /tmp/_en2.* /tmp/_list.txt
echo "wrote $(pwd)/mixed_en_he.wav"
