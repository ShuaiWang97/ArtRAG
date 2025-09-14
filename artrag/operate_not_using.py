




# async def global_query(
#     input_,
#     knowledge_graph_inst: BaseGraphStorage,
#     entities_vdb: BaseVectorStorage,
#     relationships_vdb: BaseVectorStorage,
#     text_chunks_db: BaseKVStorage[TextChunkSchema],
#     query_param: QueryParam,
#     global_config: dict,
# ) -> str:
#     context = None
#     use_model_func = global_config["llm_model_func"]
#     if isinstance(input_, dict):
#         query = input_["text"]
#         image = input_["image"]
#     else:
#         query = input_

#     kw_prompt_temp = PROMPTS["keywords_extraction"]
#     kw_prompt = kw_prompt_temp.format(query=query)
#     result = await use_model_func(kw_prompt)

#     try:
#         keywords_data = json.loads(result)
#         keywords = keywords_data.get("high_level_keywords", [])
#         keywords = ", ".join(keywords)
#     except json.JSONDecodeError:
#         try:
#             result = (
#                 result.replace(kw_prompt[:-1], "")
#                 .replace("user", "")
#                 .replace("model", "")
#                 .strip()
#             )
#             result = "{" + result.split("{")[1].split("}")[0] + "}"

#             keywords_data = json.loads(result)
#             keywords = keywords_data.get("high_level_keywords", [])
#             keywords = ", ".join(keywords)

#         except json.JSONDecodeError as e:
#             # Handle parsing error
#             print(f"JSON parsing error: {e}")
#             return PROMPTS["fail_response"]
#     if keywords:
#         context = await _build_global_query_context(
#             keywords,
#             knowledge_graph_inst,
#             entities_vdb,
#             relationships_vdb,
#             text_chunks_db,
#             query_param,
#         )

#     if query_param.only_need_context:
#         return context
#     if context is None:
#         return PROMPTS["fail_response"]

#     sys_prompt_temp = PROMPTS["rag_response"]
#     sys_prompt = sys_prompt_temp.format(
#         context_data=context, response_type=query_param.response_type
#     )
#     # Use model functions to generate response
#     response = await use_model_func(
#         query,
#         system_prompt=sys_prompt,
#         query_image_path = image,
#     )
#     if len(response) > len(sys_prompt):
#         response = (
#             response.replace(sys_prompt, "")
#             .replace("user", "")
#             .replace("model", "")
#             .replace(query, "")
#             .replace("<system>", "")
#             .replace("</system>", "")
#             .strip()
#         )

#     return response


# async def _build_global_query_context(
#     keywords,
#     knowledge_graph_inst: BaseGraphStorage,
#     entities_vdb: BaseVectorStorage,
#     relationships_vdb: BaseVectorStorage,
#     text_chunks_db: BaseKVStorage[TextChunkSchema],
#     query_param: QueryParam,
# ):
#     results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

#     if not len(results):
#         return None

#     edge_datas = await asyncio.gather(
#         *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
#     )

#     if not all([n is not None for n in edge_datas]):
#         logger.warning("Some edges are missing, maybe the storage is damaged")
#     edge_degree = await asyncio.gather(
#         *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
#     )
#     edge_datas = [
#         {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
#         for k, v, d in zip(results, edge_datas, edge_degree)
#         if v is not None
#     ]
#     edge_datas = sorted(
#         edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
#     )
#     edge_datas = truncate_list_by_token_size(
#         edge_datas,
#         key=lambda x: x["description"],
#         max_token_size=query_param.max_token_for_global_context,
#     )

#     use_entities = await _find_most_related_entities_from_relationships(
#         edge_datas, query_param, knowledge_graph_inst
#     )
#     use_text_units = await _find_related_text_unit_from_relationships(
#         edge_datas, query_param, text_chunks_db, knowledge_graph_inst
#     )
#     logger.info(
#         f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
#     )
#     relations_section_list = [
#         ["id", "source", "target", "description", "keywords", "weight", "rank"]
#     ]
#     for i, e in enumerate(edge_datas):
#         relations_section_list.append(
#             [
#                 i,
#                 e["src_id"],
#                 e["tgt_id"],
#                 e["description"],
#                 e["keywords"],
#                 e["weight"],
#                 e["rank"],
#             ]
#         )
#     relations_context = list_of_list_to_csv(relations_section_list)

#     entites_section_list = [["id", "entity", "type", "description", "rank"]]
#     for i, n in enumerate(use_entities):
#         entites_section_list.append(
#             [
#                 i,
#                 n["entity_name"],
#                 n.get("entity_type", "UNKNOWN"),
#                 n.get("description", "UNKNOWN"),
#                 n["rank"],
#             ]
#         )
#     entities_context = list_of_list_to_csv(entites_section_list)

#     text_units_section_list = [["id", "content"]]
#     for i, t in enumerate(use_text_units):
#         text_units_section_list.append([i, t["content"]])
#     text_units_context = list_of_list_to_csv(text_units_section_list)

#     return f"""
#                 -----Entities-----
#                 ```csv
#                 {entities_context}
#                 ```
#                 -----Relationships-----
#                 ```csv
#                 {relations_context}
#                 ```
#                 -----Sources-----
#                 ```csv
#                 {text_units_context}
#                 ```
#             """




# async def _find_related_text_unit_from_relationships(
#     edge_datas: list[dict],
#     query_param: QueryParam,
#     text_chunks_db: BaseKVStorage[TextChunkSchema],
#     knowledge_graph_inst: BaseGraphStorage,
# ):
#     text_units = [
#         split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
#         for dp in edge_datas
#     ]

#     all_text_units_lookup = {}

#     for index, unit_list in enumerate(text_units):
#         for c_id in unit_list:
#             if c_id not in all_text_units_lookup:
#                 all_text_units_lookup[c_id] = {
#                     "data": await text_chunks_db.get_by_id(c_id),
#                     "order": index,
#                 }

#     if any([v is None for v in all_text_units_lookup.values()]):
#         logger.warning("Text chunks are missing, maybe the storage is damaged")
#     all_text_units = [
#         {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
#     ]
#     all_text_units = sorted(all_text_units, key=lambda x: x["order"])
#     all_text_units = truncate_list_by_token_size(
#         all_text_units,
#         key=lambda x: x["data"]["content"],
#         max_token_size=query_param.max_token_for_text_unit,
#     )
#     all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]

#     return all_text_units


# async def hybrid_query(
#     input_,
#     knowledge_graph_inst: BaseGraphStorage,
#     entities_vdb: BaseVectorStorage,
#     relationships_vdb: BaseVectorStorage,
#     text_chunks_db: BaseKVStorage[TextChunkSchema],
#     query_param: QueryParam,
#     global_config: dict,
# ) -> str:
#     low_level_context = None
#     high_level_context = None
#     use_model_func = global_config["llm_model_func"]

#     if isinstance(input_, dict):
#         query = input_["text"]
#         image = input_["image"]
#     else:
#         query = input_


#     kw_prompt_temp = PROMPTS["keywords_extraction"]
#     kw_prompt = kw_prompt_temp.format(query=query)

#     result = await use_model_func(kw_prompt)
#     try:
#         keywords_data = json.loads(result)
#         hl_keywords = keywords_data.get("high_level_keywords", [])
#         ll_keywords = keywords_data.get("low_level_keywords", [])
#         hl_keywords = ", ".join(hl_keywords)
#         ll_keywords = ", ".join(ll_keywords)
#     except json.JSONDecodeError:
#         try:
#             result = (
#                 result.replace(kw_prompt[:-1], "")
#                 .replace("user", "")
#                 .replace("model", "")
#                 .strip()
#             )
#             result = "{" + result.split("{")[1].split("}")[0] + "}"

#             keywords_data = json.loads(result)
#             # hl_keywords = keywords_data.get("high_level_keywords", [])
#             # ll_keywords = keywords_data.get("low_level_keywords", [])
#             normal_keywords = keywords_data.get("keywords", [])
#             hl_keywords = ", ".join(hl_keywords)
#             ll_keywords = ", ".join(ll_keywords)
#         # Handle parsing error
#         except json.JSONDecodeError as e:
#             print(f"JSON parsing error: {e}")
#             return PROMPTS["fail_response"]

#     if ll_keywords:
#         low_level_context = await _build_local_query_context(
#             ll_keywords,
#             knowledge_graph_inst,
#             entities_vdb,
#             text_chunks_db,
#             query_param,
#         )

#     if hl_keywords:
#         high_level_context = await _build_global_query_context(
#             hl_keywords,
#             knowledge_graph_inst,
#             entities_vdb,
#             relationships_vdb,
#             text_chunks_db,
#             query_param,
#         )

#     context = combine_contexts(high_level_context, low_level_context)

#     if query_param.only_need_context:
#         return context
#     if context is None:
#         return PROMPTS["fail_response"]

#     sys_prompt_temp = PROMPTS["rag_response"]
#     sys_prompt = sys_prompt_temp.format(
#         context_data=context, response_type=query_param.response_type
#     )
#     response = await use_model_func(
#         query,
#         system_prompt=sys_prompt,
#         query_image_path = image,
#     )
#     if len(response) > len(sys_prompt):
#         response = (
#             response.replace(sys_prompt, "")
#             .replace("user", "")
#             .replace("model", "")
#             .replace(query, "")
#             .replace("<system>", "")
#             .replace("</system>", "")
#             .strip()
#         )
#     return response



# def combine_contexts(high_level_context, low_level_context):
#     # Function to extract entities, relationships, and sources from context strings

#     def extract_sections(context):
#         entities_match = re.search(
#             r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
#         )
#         relationships_match = re.search(
#             r"-----Relationships-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
#         )
#         sources_match = re.search(
#             r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
#         )

#         entities = entities_match.group(1) if entities_match else ""
#         relationships = relationships_match.group(1) if relationships_match else ""
#         sources = sources_match.group(1) if sources_match else ""

#         return entities, relationships, sources

#     # Extract sections from both contexts

#     if high_level_context is None:
#         warnings.warn(
#             "High Level context is None. Return empty High entity/relationship/source"
#         )
#         hl_entities, hl_relationships, hl_sources = "", "", ""
#     else:
#         hl_entities, hl_relationships, hl_sources = extract_sections(high_level_context)

#     if low_level_context is None:
#         warnings.warn(
#             "Low Level context is None. Return empty Low entity/relationship/source"
#         )
#         ll_entities, ll_relationships, ll_sources = "", "", ""
#     else:
#         ll_entities, ll_relationships, ll_sources = extract_sections(low_level_context)

#     # Combine and deduplicate the entities
#     combined_entities_set = set(
#         filter(None, hl_entities.strip().split("\n") + ll_entities.strip().split("\n"))
#     )
#     combined_entities = "\n".join(combined_entities_set)

#     # Combine and deduplicate the relationships
#     combined_relationships_set = set(
#         filter(
#             None,
#             hl_relationships.strip().split("\n") + ll_relationships.strip().split("\n"),
#         )
#     )
#     combined_relationships = "\n".join(combined_relationships_set)

#     # Combine and deduplicate the sources
#     combined_sources_set = set(
#         filter(None, hl_sources.strip().split("\n") + ll_sources.strip().split("\n"))
#     )
#     combined_sources = "\n".join(combined_sources_set)

#     # Format the combined context
#     return f"""
# -----Entities-----
# ```csv
# {combined_entities}
# -----Relationships-----
# {combined_relationships}
# -----Sources-----
# {combined_sources}
# """


# async def _find_most_related_text_unit_from_entities(
#     node_datas: list[dict],
#     query_param: QueryParam,
#     text_chunks_db: BaseKVStorage[TextChunkSchema],
#     knowledge_graph_inst: BaseGraphStorage,
#     top_n=5,
# ):
#     text_units = [
#         split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
#         for dp in node_datas
#     ]
#     edges = await asyncio.gather(
#         *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
#     )
#     all_one_hop_nodes = set()
#     for this_edges in edges:
#         if not this_edges:
#             continue
#         all_one_hop_nodes.update([e[1] for e in this_edges])
#     all_one_hop_nodes = list(all_one_hop_nodes)
#     all_one_hop_nodes_data = await asyncio.gather(
#         *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
#     )
#     all_one_hop_text_units_lookup = {
#         k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
#         for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
#         if v is not None
#     }
#     all_text_units_lookup = {}
#     for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
#         for c_id in this_text_units:
#             if c_id in all_text_units_lookup:
#                 continue
#             relation_counts = 0
#             for e in this_edges:
#                 if (
#                     e[1] in all_one_hop_text_units_lookup
#                     and c_id in all_one_hop_text_units_lookup[e[1]]
#                 ):
#                     relation_counts += 1
#             all_text_units_lookup[c_id] = {
#                 "data": await text_chunks_db.get_by_id(c_id),
#                 "order": index,
#                 "relation_counts": relation_counts,
#             }
#     if any([v is None for v in all_text_units_lookup.values()]):
#         logger.warning("Text chunks are missing, maybe the storage is damaged")
#     all_text_units = [
#         {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
#     ]
#     all_text_units = sorted(
#         all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
#     )
#     all_text_units = truncate_list_by_token_size(
#         all_text_units,
#         key=lambda x: x["data"]["content"],
#         max_token_size=query_param.max_token_for_text_unit,
#     )
#     all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
#     all_text_units = all_text_units[:top_n]
#     return all_text_units