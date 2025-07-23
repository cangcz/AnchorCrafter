mkdir -p ./example/results/all_frame/
mkdir -p ./example/gt/all_frame/

for video in ./example/results/*.mp4; do
    filename=$(basename "$video" .mp4)
    ffmpeg -i  "$video"   "./example/results/all_frame/${filename}_%04d.png"
done

for video in ./example/gt/*.mp4; do
    filename=$(basename "$video" .mp4)
    ffmpeg -i  "$video"   "./example/gt/all_frame/${filename}_%04d.png"
done