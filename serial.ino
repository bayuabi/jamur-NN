#include <dht.h>

dht DHT1;
dht DHT2;
#define DHT_1 11
#define DHT_2 12

#define kipasPin 6
#define mistPin 5

float kipas, mist;
void setup() {
  Serial.begin(9600);
  pinMode(kipasPin, OUTPUT);
  pinMode(mistPin, OUTPUT);

}

void loop() {
  int dht1 = DHT1.read11(DHT_1);
  int dht2 = DHT2.read11(DHT_2);
  int temp1 = DHT1.temperature;
  int hum1 = DHT1.humidity;
  int temp2 = DHT2.temperature;
  int hum2 = DHT2.humidity;

  String kirim ="";
  kirim += temp1;
  kirim += ",";
  kirim += hum1;
  kirim += ",";
  kirim += temp2;
  kirim += ",";
  kirim += hum2;
  Serial.println(kirim);
  if(Serial.available()){
    String c = Serial.readStringUntil('\n');
    int koma = c.indexOf(',');
    kipas = c.substring(0,koma).toFloat();
    mist = c.substring(koma+1, c.length()).toFloat();
    //Serial.print("Kec. Kipas: ");
    //Serial.print(kipas);
    //Serial.print("   Kec. Mis Maker: ");
    //Serial.println(mist);
  }
  kipas = 18.18*kipas - 6.36;
  mist = 10.6*mist - 19.893;
  analogWrite(kipasPin, kipas);
  analogWrite(mistPin, mist);

  temp1 = 0;
  hum1 = 0;
  temp2 = 0;
  hum2 = 0;
  delay(500);
}
