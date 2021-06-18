int LED_PIN1 = 2;
int LED_PIN2 = 3;
int data;


void setup() {
  Serial.begin(9600);
  pinMode(LED_PIN1, OUTPUT);
  pinMode(LED_PIN2, OUTPUT);
  digitalWrite(LED_PIN1, LOW);
  digitalWrite(LED_PIN2, LOW);
}

void loop() {
  while (Serial.available()){
    data = Serial.read();
  }
  
  if (data == '1'){
    digitalWrite(LED_PIN1, HIGH);
    digitalWrite(LED_PIN2, LOW);
    delay(1000);
    
  }
  else if (data == '0'){
    digitalWrite(LED_PIN1, LOW);
    digitalWrite(LED_PIN2, HIGH);
    delay(1000);
  }
  else if(data =='2'){
    digitalWrite(LED_PIN1, LOW);
    digitalWrite(LED_PIN2, LOW);
  }
}
