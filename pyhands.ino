void setup()
{
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(9600);
  Serial.setTimeout(-1);
}

void loop()
{
  long state;

  if(Serial.available() > 0)
  {
    state = Serial.parseInt();
  }
    if(state == 0)
    {
      digitalWrite(LED_BUILTIN, LOW);
    }
    
    else if(state == 1)
    {
      digitalWrite(LED_BUILTIN, HIGH);
    }

}
