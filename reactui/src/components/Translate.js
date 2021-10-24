import {useState} from 'react';
import Button from '@mui/material/Button';
import MaterialTextField from '@mui/material/TextField';


const Translate = ({translate}) => {
  const [value, setValue] = useState('');
  
  const handleChange = (event) => {
    setValue(event.target.value);
  };

  const handleClick = () => {
    translate(value);
  }
  
  return (
    <div>
      <MaterialTextField
      id="outlined-textarea"
      label="Multiline Placeholder"
      placeholder="English Text"
      multiline
      value={value}
      onChange={handleChange}
    />
      <Button variant="contained" onClick = {handleClick} >Contained</Button>
    </div>

  );
};

export default Translate;