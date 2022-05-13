docker build --rm -f "./Dockerfile" -t transformers:tagging_absa "."
docker tag transformers:tagging_absa 124.16.138.141/semeval22/transformers:tagging_absa