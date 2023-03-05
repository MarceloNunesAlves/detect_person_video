import datetime

def check_time(start_hour, end_hour):

    now = datetime.datetime.now().time()
    start = datetime.time(start_hour, 0)
    end = datetime.time(end_hour, 0)

    if start.hour > end.hour:
        if start <= now <= datetime.time(23, 59) or datetime.time(0, 0) <= now <= end:
            return True
        else:
            return False
    else:
        if start <= now <= end:
            return True
        else:
            return False

def teste():
    if check_time():
        print("A hora atual está dentro do intervalo configurado")
    else:
        print("A hora atual NÃO está dentro do intervalo configurado")