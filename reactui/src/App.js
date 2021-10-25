import { useState } from 'react';
import './App.css';
import Translate from './components/Translate';
import findClosest from './functions/findClosest';
import swuChange from './functions/towd';

function App() {
  const [translated, setTranslated] = useState('');
  const [towd, setTowd] = useState('');
  const [resultEnglish, setResultEnglish] = useState('');

  const translateToSignWriting = async (text) => {
    if(text === ''){
      setTranslated('Please Type the text');
    }
    else{
      const result = await findClosest(text);
        if(result === 'not found'){
          setTranslated('Our Data Set is limited,for now we could not found the sentence you want');
        }
        else{
          setTranslated(result[0]);
          const td = swuChange(result[0]);
          setTowd(td);
          setResultEnglish(result[1]);
        }
  }
  }

  return (
    <div className="p-3 d-flex flex-column justify-content-center align-items-center app">
      <h3 className="m-3">English to Sign Writing Translator</h3>
      <Translate translate={translateToSignWriting}/>
      <p className="m-3 font-medium " >The 1D (Formal Sign writing) Translation for word "{resultEnglish}" :</p>
      <p className="ssw-one-d font-medium ">{translated}</p>
      <p>The 2D Format:</p>
      <div className="signtext" dangerouslySetInnerHTML={{__html: towd}}></div>
    </div>
  );
  
}

export default App;
