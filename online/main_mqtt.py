import paho.mqtt.client as mqtt


class MqttConnect(object):
    def __init__(self):
        self.client = mqtt.Client(client_id="recog_device", clean_session=True, userdata=None, protocol=mqtt.MQTTv311,transport="tcp")  # client_id 수정
        self.client.username_pw_set(username="keti", password="keti")
        self.client.on_connect = self.on_connect
        self.client.on_publish = self.on_publish
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def set_client(self,client_id):
        self.client = mqtt.Client(client_id=client_id, clean_session=True, userdata=None, protocol=mqtt.MQTTv311, ##c
                                  transport="tcp")

    def user_id_pw_set(self,username,passwd):
        self.client.username_pw_set(username=username, password=passwd)

    def connect(self,ip_address,port,keep_alive):
        self.client.connect(ip_address, port, keep_alive)  # ip_adress, port 입력 전임님 확인

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("connect ok")
        else:
            print("Bad connection Returned code=", rc)

    def on_publish(self, client, userdata, mid): ##succes publish message
        print("In on_pub callback mid= ", mid)

    def on_message(self, client, userdata, msg):
        print("message Received" + str(msg.payload.decode("utf-8")))

    def on_disconnect(self, client, userdata, flags, rc=0):
        print("disconnect")

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("subscribed: " + str(mid) + " " + str(granted_qos))

#
# def on_connect(client, userdata, flags, rc):
#     if rc == 0:
#         print("connected OK")
#     else:
#         print("Bad connection Returned code=", rc)
#
#
# def on_disconnect(client, userdata, flags, rc=0):
#     print(str(rc))
#
#
# def on_publish(client, userdata, mid):
#     print("In on_pub callback mid= ", mid)
#
#
# def on_subscribe(client, userdata, mid, granted_qos):
#     print("subscribed: " + str(mid) + " " + str(granted_qos))
#
#
# def on_message(client, userdata, msg):
#     print(str(msg.payload.decode("utf-8")))