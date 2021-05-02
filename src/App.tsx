/* eslint-disable global-require */
/* eslint-disable prefer-destructuring */
import React from 'react';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import Container from '@material-ui/core/Container';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';
import './App.global.css';

const onChange = (event) => {
  const { files } = event.target;
  const arreglo = { ...files };
  const path = arreglo[0].path;
  const comando = `python src/OCR-Mike/ocr_handwriting.py --model src/OCR-Mike/handwriting.model --image ${path}`;
  const { exec } = require('child_process');
  exec(comando, (error, stdout, stderr) => {
    if (error) {
      throw error;
    }
    console.log('funcionó ezkill');
  });
};

const app = () => {
  const fileInput = React.useRef();

  return (
    <Grid container justify="center">
      <Button
        variant="contained"
        color="primary"
        onClick={() => fileInput.current.click()}
        fullWidth
      >
        Sube tu Imagen{' '}
      </Button>
      <input
        ref={fileInput}
        type="file"
        style={{ display: 'none' }}
        accept="image/x-png,image/gif,image/jpeg"
        onChange={onChange}
      />
    </Grid>
  );
};

export default function App() {
  return (
    <Router>
      <Switch>
        <Route path="/" component={app} />
      </Switch>
    </Router>
  );
}
