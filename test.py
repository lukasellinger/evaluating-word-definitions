from factscore.factscorer import CustomFactScorer

fs = CustomFactScorer(model_name=args.model_name,
                          data_dir=args.data_dir,
                          model_dir=args.model_dir,
                          cache_dir=args.cache_dir,
                          openai_key=args.openai_key,
                          cost_estimate=args.cost_estimate,
                          abstain_detection_type=args.abstain_detection_type)