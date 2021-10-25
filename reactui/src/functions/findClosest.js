import distance from './distance';
import fetchData from './fetch';

const findClosest = async (str) => {
    let index = 0 ;
    let minDis = str.length/2;
    let result = -1;
    let signWriting;
    const array = await fetchData();
    const length = array.length;
    while(index<length){
        const s1 = str.toLowerCase();
        const s2 = array[index][1].toLowerCase();
        if(s1 === s2){
            result = index;
        }
        index++;
    }
    if(result === -1){
      index = 0;
      while(index<length){
        if(!((array[index][1].length < (str.length/2)) || (str.length < (array[index][1].length/2)))){
          const dis = distance(str, array[index][1]);
          if(dis < minDis) {
              result = index;
          }
        }
        index++;
      }
    }
    if(result===-1){
        signWriting = 'not found';
    }
    else{
        signWriting = [];
        signWriting[0] = array[result][0]; 
        signWriting[1] = array[result][1]; 
    }
    return signWriting;
}
export default findClosest;