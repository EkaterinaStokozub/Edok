pip install telebot
import telebot
from telebot import types  # Импортируем типы для кнопок
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk.corpus import stopwords

# Загрузка данных
recipes = pd.read_csv('/content/drive/MyDrive/Edok/povarenok.csv')

# Просмотр первых строк данных
print("Первые строки данных:")
print(recipes.head())

# Проверка наличия пропущенных значений
print("\nПропущенные значения по колонкам:")
print(recipes.isnull().sum())

# Удаление строк с пропущенными значениями в ключевых столбцах
recipes = recipes.dropna(subset=['name', 'ingredients'])

# Проверка и удаление дубликатов по полю 'url'
print("\nКоличество дубликатов по полю 'url':")
print(recipes.duplicated(subset=['url']).sum())
recipes = recipes.drop_duplicates(subset=['url'])

# Проверка и удаление дубликатов по полю 'name'
print("\nКоличество дубликатов по полю 'name':")
print(recipes.duplicated(subset=['name']).sum())
recipes = recipes.drop_duplicates(subset=['name'])

# Проверка уникальности значений в поле 'name'
print("\nУникальные значения в поле 'name':")
print(recipes['name'].nunique())

# Проверка уникальности значений в поле 'url'
print("\nУникальные значения в поле 'url':")
print(recipes['url'].nunique())

# Подготовка списка русских стоп-слов
nltk.download('stopwords')
russian_stopwords = stopwords.words("russian")

# Дополнительная обработка текстовых данных (например, ингредиентов)
recipes['name'] = recipes['name'].str.lower().str.strip()
recipes['ingredients'] = recipes['ingredients'].str.lower().str.strip()

# Повторная проверка дубликатов после нормализации названий
print("\nКоличество дубликатов по полю 'name' после нормализации:")
print(recipes.duplicated(subset=['name']).sum())
print("\nКоличество дубликатов по полю 'url' после нормализации:")
print(recipes.duplicated(subset=['url']).sum())

# Векторизация ингредиентов
vectorizer = TfidfVectorizer(stop_words=russian_stopwords)
ingredients_tfidf = vectorizer.fit_transform(recipes['ingredients'].fillna(''))

# Функция для получения рекомендаций на основе введенных ингредиентов
def get_recommendations(ingredients):
    ingredients_str = ' '.join(ingredients)
    query_vec = vectorizer.transform([ingredients_str])
    cosine_similarities = linear_kernel(query_vec, ingredients_tfidf).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1]  # Сортировка по убыванию
    recommended_recipes = recipes.iloc[related_docs_indices][:5].reset_index(drop=True)  # Выбираем топ-5 рецептов и сбрасываем индексы
    return recommended_recipes[['name', 'ingredients', 'url']]


ingredients_example = ['мясо', 'картошка', 'лук']
recommended_recipes_example = get_recommendations(ingredients_example)
print("\nРекомендованные рецепты:")
print(recommended_recipes_example)

ingredients_example = ['банан', 'мука', 'сметана']
recommended_recipes_example = get_recommendations(ingredients_example)
print("\nРекомендованные рецепты:")
print(recommended_recipes_example)

# Инициализация бота
API_TOKEN = '7093711934:AAG3GGRAX5WYYu3JtpJisDr9e9ApsJuD9RM'
bot = telebot.TeleBot(API_TOKEN)

user_data = {}

# Функция для обработки команды /start и начального приветствия
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я помогу тебе найти рецепт. Какие у тебя есть ингредиенты? Перечисли их через запятую.")

# Функция для обработки сообщений с ингредиентами и отправки рекомендаций
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_ingredients = [ingredient.strip() for ingredient in message.text.split(',')]
    recommended_recipes = get_recommendations(user_ingredients)
    user_data[message.chat.id] = {
        'state': 'awaiting_selection',
        'recommended_recipes': recommended_recipes
    }
    response = "Вот несколько рецептов, которые я нашел для тебя:\n\n"
    keyboard = types.InlineKeyboardMarkup(row_width=1)  # Создаем клавиатуру с одной кнопкой в строке
    for index, row in recommended_recipes.iterrows():
        button_text = row['name']
        callback_data = f"recipe_{index}"  # Уникальный идентификатор для каждой кнопки
        keyboard.add(types.InlineKeyboardButton(text=button_text, callback_data=callback_data))
    response += "Выбери рецепт, который тебе нравится:"
    bot.send_message(message.chat.id, response, reply_markup=keyboard)

# Функция для обработки выбора рецепта через кнопку
@bot.callback_query_handler(func=lambda call: call.message.chat.id in user_data and user_data[call.message.chat.id]['state'] == 'awaiting_selection')
def callback_handler(call):
    recipe_index = int(call.data.split('_')[1])  # Извлекаем индекс рецепта из callback_data
    user_info = user_data[call.message.chat.id]
    recommended_recipes = user_info['recommended_recipes']
    if 0 <= recipe_index < len(recommended_recipes):
        recipe = recommended_recipes.iloc[recipe_index]
        response = f"Название рецепта: {recipe['name']}\n\n"
        response += f"Ингредиенты: {recipe['ingredients']}\n\n"
        response += f"Полный рецепт можно найти здесь: {recipe['url']}"
        bot.send_message(call.message.chat.id, response)
        # Обновляем состояние пользователя после отправки рецепта
        user_data[call.message.chat.id]['state'] = 'awaiting_ingredients'
    else:
        bot.send_message(call.message.chat.id, "Неверный номер рецепта. Пожалуйста, выбери правильный номер.")

# Запуск бота
bot.polling()
