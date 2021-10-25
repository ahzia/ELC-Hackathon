import {useState} from 'react';
import LoadingButton from '@mui/lab/LoadingButton';
import MaterialTextField from '@mui/material/TextField';


const Translate = ({translate}) => {
  const [value, setValue] = useState('');
  const [loading, setLoading] = useState(false);
  
  const handleChange = (event) => {
    setValue(event.target.value);
  };

  const handleClick = () => {
    setLoading(true);
    translate(value).then(() => {
      setLoading(false);
    });
  }
  
  return (
    <div className="d-flex flex-column align-items-center">
      <MaterialTextField
      id="outlined-textarea"
      label="English Text"
      placeholder="English Text"
      multiline
      value={value}
      onChange={handleChange}
    />
    <br />
    <LoadingButton variant="contained" loading={loading} onClick ={handleClick} >Translate</LoadingButton>
    </div>

  );
};

export default Translate;