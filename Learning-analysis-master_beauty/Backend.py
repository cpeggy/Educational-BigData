# process_and_graph.py
from flask import Flask, request, render_template, Response
import pandas as pd
from flask import jsonify
import os
import json
import requests
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import RecognizeCustomEntitiesAction
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')

app = Flask(__name__, static_url_path='/static', static_folder='templates')

UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Azure Text Analytics API 配置
sentiment_endpoint = "https://nertraining.cognitiveservices.azure.com/text/analytics/v3.0/sentiment"
sentiment_key = "b0d806bde46146c99a8909e0aa1e9757"

learning_method_endpoint = "https://nerrrrr.cognitiveservices.azure.com/"
learning_method_key = "930ae92c8d65431cbae23c981d3881f4"

subject_endpoint = "https://nerrrrr.cognitiveservices.azure.com/"
subject_key = "930ae92c8d65431cbae23c981d3881f4"

@app.route('/')
def index():
    return render_template('index.html')

UPLOAD_FOLDER = 'uploads'

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']

    if uploaded_file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)

        df = pd.read_csv(file_path)


        dataresult = []
        for i in range(len(df.iloc[:, 0])):
            subdic = {
                "course": df.iloc[i, 0],
                "score": int(df.iloc[i, 1]),
                "review": df.iloc[i, 2]
            }
            dataresult.append(subdic)

        review_data = [item["review"] for item in dataresult]
        course_data = [item["course"] for item in dataresult]

        sentiment_results = []
        learning_method_results = []
        subject_field_results = []

        for review_text in review_data:
            # 使用Sentiment分析的配置
            sentiment_response = analyze_sentiment(sentiment_endpoint, sentiment_key, review_text)
            if sentiment_response and 'documents' in sentiment_response:
                sentiment = sentiment_response['documents'][0].get('sentiment')
                sentiment_results.append({"sentiment": sentiment})
            else:
                sentiment_results.append({"sentiment": "N/A"})

        for review_text in review_data:
            # 使用Learning Method分析的配置
            learning_method_response = analyze_learning_method(learning_method_endpoint, learning_method_key, [review_text])
            if 'error' not in learning_method_response:
                result = learning_method_response.get("learning method", [])
                if result:
                    learning_method_results.append({"learning_method": result[0]["text"]})
                else:
                    learning_method_results.append({"learning_method": "N/A"})
            else:
                learning_method_results.append({"learning_method": "N/A"})

        for course_text in course_data:
            # 使用Subject分析的配置
            subject_response = analyze_subject(subject_endpoint, subject_key, [course_text])
            if 'error' not in subject_response:
                result = subject_response.get("field", [])
                if result:
                    subject_field_results.append({"field": result[0]["category"]})
                else:
                    subject_field_results.append({"field": "N/A"})
            else:
                subject_field_results.append({"field": "N/A"})

        for i in range(len(dataresult)):
            dataresult[i]["sentiment"] = sentiment_results[i]["sentiment"]
            dataresult[i]["learning_method"] = learning_method_results[i]["learning_method"]
            dataresult[i]["field"] = subject_field_results[i]["field"]

        # 將分析後的結果寫入json檔
        json_file_path = 'result.json'
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(dataresult, json_file, ensure_ascii=False)

        # 進行圖表生成
        graph(json_file_path)

        # 使用 json.dumps 手動編碼 JSON 數據，並設置 ensure_ascii 為 False
        json_data = json.dumps(dataresult, ensure_ascii=False)

        # 渲染新的HTML模板，並將JSON數據傳遞給模板
        return jsonify(result=dataresult, success=True)

    return jsonify(error="文件上传成功。")


# Sentiment分析的函数
def analyze_sentiment(sentiment_endpoint , sentiment_key, text):
    sentiment_analysis_api_url= sentiment_endpoint

    headers = {
        "Ocp-Apim-Subscription-Key": sentiment_key,
        "Content-Type": "application/json",
    }

    data = {
        "documents": [
            {
                "language": "en",
                "id": "1",
                "text": text,
            }
        ]
    }

    response = requests.post(sentiment_analysis_api_url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Sentiment analysis failed"}

# Learning Method分析的函数
def analyze_learning_method(endpoint, key, text_list):
    project_name = "nerr202310231" 
    deployment_name = "nerlearningmethodDeployment_version1"

    text_analytics_client = TextAnalyticsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    response_data = {"learning method": []}

    for text in text_list:
        poller = text_analytics_client.begin_analyze_actions(
            [text],
            actions=[
                RecognizeCustomEntitiesAction(
                    project_name=project_name,
                    deployment_name=deployment_name
                ),
            ],
        )

        document_results = poller.result()

        for result in document_results:
            custom_entities_result = result[0]  # first document, first result
            if not custom_entities_result.is_error:
                # 解析 custom_entities_result 並將其添加到 response_data
                for entity in custom_entities_result.entities:
                    entity_data = {
                        'text': entity.text,
                        'confidence_score': entity.confidence_score
                    }
                    response_data["learning method"].append(entity_data)
            else:
                response_data["learning method"].append({"error": "N/A"})

    return response_data

# Subject分析的函数
def analyze_subject(endpoint, key, text_list):
    project_name = "nersubject" 
    deployment_name = "ner_subjectverson2"

    text_analytics_client = TextAnalyticsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    response_data = {"field": []}

    for text in text_list:
        poller = text_analytics_client.begin_analyze_actions(
            [text],
            actions=[
                RecognizeCustomEntitiesAction(
                    project_name=project_name,
                    deployment_name=deployment_name
                ),
            ],
        )

        document_results = poller.result()

        for result in document_results:
            custom_entities_result = result[0]  # first document, first result
            if not custom_entities_result.is_error:
                # 解析 custom_entities_result 並將其添加到 response_data
                for entity in custom_entities_result.entities:
                    entity_data = {
                        'category': entity.category,
                        'confidence_score': entity.confidence_score
                    }
                    response_data["field"].append(entity_data)
            else:
                response_data["field"].append({"error": "N/A"})

    return response_data

# 生成圖表的函數
def graph(json_file_path):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    # JSON data
    df = pd.read_json(json_file_path)
    df1 = df.copy()
    df2 = df.copy()

    # 圖1, 總整圖
    category_colors = {
        "語文": '#C6D7E3',
        "數理": '#D3DBC5',
        "藝術": '#E4D2BB',
        "社會": '#D5C6D9',
        "體育": '#EDE5BF',
    }
    # Create the horizontal bar chart with colored bars
    plt.rcParams["font.family"] = "Microsoft JhengHei"
    plt.figure(figsize=(10, 6))
    for category, color in category_colors.items():
        category_data = df[df['field'] == category]
        if not category_data.empty:  # Check if data exists for the category
            bars = plt.barh(category_data['course'], category_data['score'], color=color, label=category)
            for bar, sent, method in zip(bars, category_data['sentiment'], category_data['learning_method']):
                plt.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f'Sentiment: {sent}, Method: {method}',
                    va='center',
                    ha='right',
                )
    plt.xlabel('Score')
    plt.ylabel('Course')
    plt.title('Scores by Course with Sentiment & Learning Method Analysis')
    plt.legend(loc='lower right')
    plt.gca().invert_yaxis()  # Invert the y-axis for better readability
    plt.savefig('static/graph1.png')  # Save the figure as an image

    # 圖2、3，類別平均成績與情緒分析
    df1 = df
    meanscore_cate = df1.groupby("field")["score"].mean().round(1)
    df1["sentiment"] = pd.Categorical(df1["sentiment"], categories=["negative", "neutral", "mixed", "positive"]).codes
    meansentiment = df1.groupby("field")["sentiment"].mean().round(1)
    # category
    cat_data = df1.groupby("field")
    cat = []
    for i in cat_data:
        cat.append(i[0])
    data_for_graph = pd.DataFrame(
        {"Field": cat, "Mean Score": meanscore_cate.values, "Mean Sentiment": meansentiment.values}
    )
    # 最後建議的學科領域類別
    max_score_value = data_for_graph['Mean Score'].max()
    max_sentiment_score_value = data_for_graph["Mean Sentiment"].max()
    max_score_fields = data_for_graph.loc[data_for_graph['Mean Score'] == max_score_value, 'Field'].tolist()
    max_sentiment_Score_field = data_for_graph.loc[
        data_for_graph["Mean Sentiment"] == max_sentiment_score_value, "Field"
    ].tolist()

    category_colors = {
        "語文": '#C6D7E3',
        "數理": '#D3DBC5',
        "藝術": '#E4D2BB',
        "社會": '#D5C6D9',
        "體育": '#EDE5BF',
    }
    # Apply the custom color palette to data_for_graph DataFrame
    data_for_graph["color"] = data_for_graph["Field"].map(category_colors)
    # mean score 圖
    plt.figure(figsize=(10, 6))
    # 將這行代碼的 palette 改為 hue，同時添加 legend=False
    sns.barplot(data=data_for_graph, x="Field", y="Mean Score", palette=list(data_for_graph["color"]), legend=False)
    plt.title("Mean Score by Field")
    plt.savefig('static/graph2.png')  # Save the figure as an image
    # mean sentiment 圖
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data_for_graph, x="Field", y="Mean Sentiment", palette=list(data_for_graph["color"]), legend=False)
    plt.title("Mean Sentiment by Field")
    plt.savefig('static/graph3.png')  # Save the figure as an image

    # 圖4，正向情緒的學習方式成績圖
    # 處理na的資料
    method_data = df2[df2["learning_method"] != "N/A"]
    sentiment_is_positive = method_data[method_data["sentiment"] == "positive"].drop(["review"], axis=1)
    # 將一樣的學習方式分數取平均(可能之後整理出來的資料不會有機會重複)
    sentiment_for_method = sentiment_is_positive.groupby("learning_method")["score"].mean()
    # method
    method_data = sentiment_is_positive.groupby("learning_method")
    method = []
    for i in method_data:
        method.append(i[0])
    # method data 整理
    method_for_graph = pd.DataFrame({"learning_method": method, "mean_score": sentiment_for_method.values})
    # 按照分数高低排序
    method_for_graph = method_for_graph.sort_values(by='mean_score', ascending=False)
    # Find the learning method with the highest score
    max_score_method = method_for_graph["mean_score"].max()
    max_score_method = method_for_graph.loc[
        method_for_graph['mean_score'] == max_score_method, 'learning_method'
    ].tolist()
    # Create the horizontal bar chart with colored bars
    plt.rcParams["font.family"] = "Microsoft JhengHei"
    plt.figure(figsize=(10, 6))
    colors = ['#e0938d' if method in max_score_method else '#E2C6C4' for method in method_for_graph['learning_method']]
    plt.barh(method_for_graph["learning_method"], method_for_graph['mean_score'], color=colors)
    plt.xlabel('Score')
    plt.ylabel('Learning Method')
    plt.title('Learning Method with Score (positive sentiment only)')
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.savefig('static/graph4.png')  # Save the figure as an image
    
# 調用 graph 函數，傳入 JSON 文件的路徑
# graph("result.json")

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify(error=str(e)), 500

def recom(jsondata):
    import pandas as pd
    import json
    # JSON data
    jsondata = pd.DataFrame(jsondata)

    df1=jsondata.copy()
    df2=jsondata.copy()

    #類別平均成績與情緒分析
    df1 = jsondata
    meanscore_cate = df1.groupby("field")["score"].mean().round(1)
    df1["sentiment"] = pd.Categorical(df1["sentiment"], categories=["negative", "neutral", "mixed", "positive"]).codes
    meansentiment = df1.groupby("field")["sentiment"].mean().round(1)

    # category
    cat_data = df1.groupby("field")
    cat = []
    for i in cat_data:
        cat.append(i[0])

    data_for_graph = pd.DataFrame({"Field": cat, "Mean Score": meanscore_cate.values, "Mean Sentiment": meansentiment.values})

    # 最後建議的學科領域類別
    max_score_value = data_for_graph['Mean Score'].max()
    max_sentiment_score_value = data_for_graph["Mean Sentiment"].max()
    max_score_fields = data_for_graph.loc[data_for_graph['Mean Score'] == max_score_value, 'Field'].tolist()
    max_sentiment_Score_field = data_for_graph.loc[
        data_for_graph["Mean Sentiment"] == max_sentiment_score_value, "Field"].tolist()

    # 正向情緒的學習方式成績圖
    # 處理na的資料
    method_data = df2[df2["learning_method"] != "N/A"]
    sentiment_is_positive = method_data[method_data["sentiment"] == "positive"].drop(["review"], axis=1)

    # 將一樣的學習方式分數取平均(可能之後整理出來的資料不會有機會重複)
    sentiment_for_method = sentiment_is_positive.groupby("learning_method")["score"].mean()

    # method
    method_data = sentiment_is_positive.groupby("learning_method")
    method = []
    for i in method_data:
        method.append(i[0])

    # method data 整理
    method_for_graph = pd.DataFrame({"learning_method": method, "mean_score": sentiment_for_method.values})

    # 按照分数高低排序
    method_for_graph = method_for_graph.sort_values(by='mean_score', ascending=False)

    # Find the learning method with the highest score
    max_score_method = method_for_graph["mean_score"].max()
    max_score_method = method_for_graph.loc[method_for_graph['mean_score'] == max_score_method, 'learning_method'].tolist()

    # 學習建議輸出
    if len(max_score_fields) == 1 and len(max_sentiment_Score_field) == 1:
        if max_score_fields[0] == max_sentiment_Score_field[0]:
            result = (f'透過您上傳的資料我們發現您對於「{max_sentiment_Score_field[0]}」領域的情感表現最為正向，'
                      f'而就整體平均分數而言您在「{max_score_fields[0]}」領域表現得最好，您或許對於「{max_sentiment_Score_field[0]}」領域具有強大的興趣與學習優勢，'
                      f'因此您可以考慮以「{max_score_fields[0]}」領域方向作為學習發展的目標!\n')
            if len(max_score_method) == 1:
                result = result + f'''根據資料顯示，您最佳的學習方式為「{max_score_method[0]}」。或許未來在學習不同科目時，您可以選擇此種學習方式提升學習效果。'''
                return result
            else:
                inputmethod = ""
                for i in max_score_fields:
                    inputmethod += f'{i} '
                result = result + f'''根據資料顯示，您最佳的學習方式為{inputmethod}。或許未來在學習不同科目時，您可以選擇這些學習方式提升學習效果。'''
                return result
        else:
            result = (f'透過您上傳的資料我們發現您對於「{max_sentiment_Score_field[0]}」的情感表現最為正向，'
                      f'而就整體平均分數而言您在「{max_score_fields[0]}」表現得最好，您或許對於「{max_sentiment_Score_field[0]}」具有強大的興趣，'
                      f'且您在「{max_score_fields[0]}」具有學習優勢。若您可以將這兩個科目結合做為未來的學習方向發展，或許會有不錯的成果!\n')
            if len(max_score_method) == 1:
                result = result + f"根據資料顯示，您最佳的學習方式為「{max_score_method[0]}」。或許未來在學習不同科目時，您可以選擇此種學習方式提升學習效果。"
                return result
            else:
                inputmethod = ""
                for i in max_score_fields:
                    inputmethod += f'{i}、'
                inputmethod = inputmethod[:len(inputmethod)]
                result = result + f"根據資料顯示，您最佳的學習方式為「{inputmethod}」。或許未來在學習不同科目時，您可以選擇這些學習方式提升學習效果。"
                return


@app.route('/check_for_new_images', methods=['GET'])
def check_for_new_images():
    image_files = ['graph1.png', 'graph2.png', 'graph3.png', 'graph4.png']
    new_images = [image for image in image_files if os.path.exists(os.path.join('static', image))]
    
    # 將 new_images 按照指定順序排序
    sorted_new_images = sorted(new_images, key=lambda x: image_files.index(x))
    
    return Response(response=json.dumps({'new_images': sorted_new_images}),
                    status=200,
                    mimetype='application/json')

@app.route('/recommendation', methods=['POST'])
def get_recommendation():
    jsondata = request.get_json()  # 從前端接收 JSON 數據
    recommendation = recom(jsondata)  # 調用你的文字建議函數
    return jsonify({'recommendation': recommendation})


if __name__ == '__main__':
    app.run(debug=True)