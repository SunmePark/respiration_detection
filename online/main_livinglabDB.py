import time
# import datetime
import pymysql
from datetime import datetime, timedelta

# class livinglabDB():
#     # def __init__(self):
#     #     self.user = 'hcrweb'
#     #     self.passwd = 'keti1234'
#     #     self.host = '166.104.60.178'
#     #     self.port = 28798
#     #     self.db = 'hcr_apps_db'
#     #     self.charset = 'utf8'
#
#
#     def __init__(self):
#         self.user = 'root'
#         self.passwd = '1'
#         self.host = "localhost"
#         self.port = 3306
#         self.db = 'test'
#         self.livingdb = None
#         self.charset = 'utf8mb4'
#
#
#     def DB_connect(self):
#         try:
#             self.livingdb = pymysql.connect(
#                 user=self.user,
#                 passwd=self.passwd,
#                 host=self.passwd,
#                 port=self.port,
#                 db=self.db,
#                 charset=self.charset,
#             )
#             print("DB connected")
#
#         except Exception as e:
#             print(e)
#             time.sleep(1)
#             self.DB_connect()



def db_connect():
    try:
        # livinglab_db = pymysql.connect(
        #     user='root',
        #     passwd='1234',
        #     host="localhost",
        #     port=3306,
        #     db='test',
        #     charset='utf8mb4'
        # )

        livinglab_db = pymysql.connect(
            user='hcrweb',
            passwd='keti1234',
            host='166.104.60.178',
            port=28798,
            db='hcr_apps_db',
            charset='utf8',
        )
        print("DB connected")

    except Exception as e:
        print(e)

    return livinglab_db



# def rawdatatoDB(livingdb, val):
#     # for k in range(len(a_pd)):
#     sql_dict = dict(datetimes=time.strftime('%Y-%m-%d %X', time.localtime(time.time())),
#                     value=val)
#
#     sql = "INSERT INTO `respiration_data2` ( %s ) VALUES  %s " % (
#         ', '.join(sql_dict.keys()), tuple(sql_dict.values()))
#
#     cursor = livingdb.cursor()
#     cursor.execute(sql)
#     livingdb.commit()
#
#     return


def resulttoDB(livingdb, res):

    sql_dict = dict(datetimes=(datetime.now() - timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'),
                    result=res)

    sql = "INSERT IGNORE INTO `respiration_result` ( %s ) VALUES  %s " % (
        ', '.join(sql_dict.keys()), tuple(sql_dict.values()))


    # b = [(datetime.now() - timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S')]
    # b.append(res)
    #
    # sql = "INSERT INTO `respiration_result` VALUES" + str(tuple(b))


    cursor = livingdb.cursor()
    cursor.execute(sql)
    livingdb.commit()

    return



# def signaltoDB(livingdb, signal):
#     b = [time.strftime('%Y-%m-%d %X', time.localtime(time.time()))]
#     b.extend(signal)
#
#     sql = "INSERT INTO `respiration_signal` VALUES" + str(tuple(b))
#
#     cursor = livingdb.cursor()
#     cursor.execute(sql)
#     livingdb.commit()
#
#     return



def signaltoDB(livingdb, amplitude):

    sql_dict = dict(datetimes=(datetime.now() - timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'),
                    amplitude=amplitude)

    sql = "INSERT IGNORE INTO `respiration_signal` ( %s ) VALUES  %s " % (
        ', '.join(sql_dict.keys()), tuple(sql_dict.values()))

    # b = [(datetime.now() - timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S')]
    # b.append(amplitude)
    #
    # sql = "INSERT INTO `respiration_signal` VALUES" + str(tuple(b))

    cursor = livingdb.cursor()
    cursor.execute(sql)
    livingdb.commit()
    # print('upload success!')

    return


#
# def radartoDB(livingdb, signal):
#     b = [(datetime.now() - timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S')]
#     b.extend(signal)
#
#     sql = "INSERT INTO `respiration_radar` VALUES" + str(tuple(b))
#
#     cursor = livingdb.cursor()
#     cursor.execute(sql)
#     livingdb.commit()
#
#     return


"""
livingdb = db_connect()
key = ''
for i in range(64):
    str_key = 's'+str(i)+' FLOAT,'
    key = key + str_key

sql = "CREATE TABLE respiration_signal (datetimes DATETIME, " + key[:-1] + ")"

cursor = livingdb.cursor()
cursor.execute(sql)
"""




