python -m sglang.launch_server \
    --model-path intfloat/e5-base-v2 \
    --is-embedding \
    --tp 4 \
    --mem-fraction-static 0.1 \
    --log-level warning &

python envs/local_search_service.py \
    --model_name intfloat/e5-base-v2 \
    --index_path /vast/llm/andy/search-r1/e5_Flat.index \
    --corpus_path /vast/llm/andy/search-r1/wiki-18.jsonl \
    --top_k 3 &

while [ $(curl -s -o /dev/null -w "%{http_code}" http://localhost:30000/health) -ne 200 ]; do
    sleep 1
done

while [ $(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health) -ne 200 ]; do
    sleep 1
done