# OPTIMISING A SMALL LLM Using WANDA and QUANTIZING IT TO 4-bits
    This is my take at optimisinig an llm. The current plan 
    is to take up a small openweight llm and fine-tune it using lora
    then apply WANDA to get a sparce network and also quantise it to 
    build, what i think, is going to be the absolutly efficient local llm.

    The use case of the llm we create now i abstract. But i think if this performs well
    in benchmarks, i can finetune and sparce out a much larger model for a better domain-
    specific application.

## Game-Plan
1. Decide on LLm - qwen 2.5 3b-instruct
2. Fine tune using LORA
3. Apply WANDA
4. Benchmark against base
5. Apply quantisation
6. Benchmark against base and WANDA-ed version


        __NOTE__ :
        this is going to be a story type documentation where i am going to try and document my opinions and 
        things i learn and other important aspect during this project.

## Fine Tuning
i am deciding to do a lora finetuning coz