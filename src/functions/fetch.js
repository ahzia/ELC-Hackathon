import raw from './data.txt';


const fetchData = async () => {
    const result=[];
    await fetch(raw)
    .then(r => r.text())
    .then(text => {
      var lines = text.split(/\r?\n/);
      lines.forEach(sentence => {
        var pair = sentence.split(/\t¿/);
        result.push(pair);
      })
    });
    return result;
}


export default fetchData;