import speech_recognition as sr, pyttsx3

def speech_to_text(timeout=5, phrase_time_limit=10):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        return r.recognize_google(audio)
    except Exception as e:
        return ''

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
