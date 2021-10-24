import distance from './distance';

const findClosest = (str, array) => {
    let index = 0 ;
    let minDis = str.length/2;
    let result = -1;
    array.forEach(element => {
        const dis = distance(str, element[1]);
        if(dis < minDis) {
            result = index;
        }
        index++;
    });
    return result;
}
export default findClosest;