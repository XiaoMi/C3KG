export CUDA_VISIBLE_DEVICES="2"

python construct/SBert_match.py
python construct/extract_entity.py
python construct/construct_edge.py
