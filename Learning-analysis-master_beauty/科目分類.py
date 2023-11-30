
def field(jsondata):
    import pandas as pd
    df = pd.read_json(jsondata, encoding='utf-8')
    grouped=df.groupby("field") 
    for field_value, group_df in grouped:
        json_result = group_df.to_json(indent=2, orient='records', force_ascii=False)
        print(field_value,json_result)
        print("---------------------")

field("result.json")