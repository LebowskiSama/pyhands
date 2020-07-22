int state;

void setup()
{
  pinMode(11, OUTPUT);
  Serial.begin(9600);
}

void loop()
{

  if(Serial.available())
  {
    state = Serial.parseInt();
    
    switch(state)
    {
      case 0:
        digitalWrite(11, LOW);
        break;

      case 1:
        digitalWrite(11, HIGH);
        break;

      case 2:
        for(int i=0; i < 5; i++)
        {
          digitalWrite(11, LOW);
          delay(100);
          digitalWrite(11, HIGH);
          delay(100);
        }
          break;

      case 3:
        for(int i=0; i<3; i++)
        {
          for(int j=0; j < 255; j++)
          {
            analogWrite(11, j);
            delay(3);
          }
          delay(3);
          for(int k=255; k > 0; k--)
          {
            analogWrite(11, k);
            delay(3);
          }
        }
          break;
      
    }
  }
  delay(1);
}
