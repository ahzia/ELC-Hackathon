import { useState } from 'react';
import './App.css';
import Translate from './components/Translate';
import fetchData from './functions/fetch';
import findClosest from './functions/findClosest';

function App() {
  const [result, setResult] = useState([]);
  const [translated, setTranslated] = useState('');
  fetchData().then(res => {
    setResult(res);
  });
  const translate = (text) => {
    if(text === ''){
      setTranslated('Please Type the text');
    }
    else{
      const index = findClosest(text, result);
        if(index === -1){
          setTranslated('Our Data Set is limited, we could not found the sentence you want');
        }
        else{
          setTranslated(result[index][0]);
        }
  }
  }
  if(result === []){
    <p>Waiting</p>
  }
  return (
    <div>
      <Translate translate={translate}/>
      <p>{translated}</p>
    </div>
  );
  
}

export default App;
