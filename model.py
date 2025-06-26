import pickle
import json

import pandas as pd

from utils.settings import MODEL_PATH


class ModelObject:
    """
    Класс для предсказания риска сердечных приступов
    """
    def __init__(self):
        super().__init__()
        self.model = None
        try:
            with open(MODEL_PATH, 'rb') as rf:
                self.model = pickle.load(rf)
        except Exception:
            pass

    def __feature_engineering(self, df):
        """
        Создание новых признаков на основе имеющихся данных
        """
        df = df.copy()
        
        # Высокое давление (нормализованное значение > 0.6 примерно соответствует 140+ мм рт.ст.)
        df['hypertension'] = ((df['systolic_blood_pressure'] > 0.6) | (df['diastolic_blood_pressure'] > 0.6)).astype(int)

        # Пульсовое давление (разница между верхним(систолическим) и нижним(диастолическое))
        df['pulse_pressure'] = abs(df['systolic_blood_pressure'] - df['diastolic_blood_pressure'])
        
        # Хронические заболевания (комбинация факторов риска)
        metabolic_risk_features = ['obesity', 'diabetes', 'hypertension']
        df['metabolic_risk_score'] = df[metabolic_risk_features].sum(axis=1)

        # Малоподвижный образ жизни
        df['high_sedentary'] = (df['sedentary_hours_per_day'] > 0.7).astype(int)
        df['low_exercise'] = (df['exercise_hours_per_week'] < 0.3).astype(int)
        
        # Вредные привычки и образ жизни
        lifestyle_risk_features = ['smoking', 'alcohol_consumption', 'high_sedentary', 'low_exercise']
        df['lifestyle_risk_score'] = df[lifestyle_risk_features].sum(axis=1)

        return df
    
    def __prepare_data(self, data):
        '''
        Подготавливает данные тестовой выборки для получения предсказаний
        '''
        data.columns = data.columns.str.lower().str.replace(' ', '_')
        empty_ids = data[data.isna().any(axis=1)]['id'].to_frame()
        empty_ids['prediction'] = -100
        data = data.drop(['unnamed:_0'] , axis=1).set_index('id')
        data = data.dropna()

        columns_change_type = ['diabetes', 'family_history', 'smoking', 'obesity', 'alcohol_consumption', 'previous_heart_problems', \
                            'medication_use', 'stress_level', 'physical_activity_days_per_week']
        
        data[columns_change_type] = data[columns_change_type].astype(int)
        data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
        data['heart_rate'] = data['heart_rate'] * 10
        data = data.drop(['ck-mb', 'troponin'], axis=1)
        data = self.__feature_engineering(data)
        data = data.drop(['blood_sugar', 'income', 'systolic_blood_pressure', 'diastolic_blood_pressure'], axis=1)
        
        return data, empty_ids

    def __read_data(self, data_path):
        try:
            df = pd.read_csv(data_path)
        except Exception:
            df = None
        return df
    
    def __prepare_result(self, df):
        try:
            if df is not None:
                result, df_na = self.__prepare_data(df)
            else:
                result, df_na = None
            if df_na is None:
                df_na = pd.DataFrame(columns = ['id', 'prediction'])
        except Exception:
            result, df_na = None
        return result, df_na

    def prediction(self, data_path):
        '''
        Функция для получения предсказаний, получает на вход путь к файлу с данными
        На выходе получаем json следующего вида:
        {
            "result": "success"/"failed",
            "message": <сообщение об ошибке>,
            "data": [{"id": <id>,"prediction":<prediction>},...]
            где prediction - "низкий риск"/"высокий риск"/"недостаточно данных для предсказания"
        }
        '''

        best_threshold = 0.46
        message = ""
        result = "success"

        data = self.__read_data(data_path)
        if data is None:
            result = "failed"
            message = "убедитесь, что отправляемый файл в формате csv"
            return {"result": result, "message": message, data: []}
        df, df_na = self.__prepare_result(data)
        if df is None:
            result = "failed"
            message = "некорректные данные, убедитесь, что отправляемый файл содержит все необходимые признаки"
            return {"result": result, "message": message, data: []}
        
        try:
            df['prediction'] = (self.model.predict_proba(df)[:, 1] >= best_threshold).astype(int)
            df = df['prediction'].reset_index()
            predictions = pd.concat([df, df_na])
            predictions['prediction'] = predictions['prediction'].replace(
                {
                    -100: "недостаточно данных для предсказания",
                    0: "низкий риск",
                    1: "высокий риск"
                })
            return {
                "result": result,
                "message": message,
                "data": json.loads(predictions.to_json(orient="records"))}
        except Exception:
            result = "failed"
            message = "некорректные данные, убедитесь, что отправляемый файл содержит необходимое количество признаков"
            return {
                "result": result,
                "message": message,
                "data": []
            }        