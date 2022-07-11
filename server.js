const express = require('express');
const cors = require('cors');
const tf = require('@tensorflow/tfjs');
const tfn = require('@tensorflow/tfjs-node');
const fetch = (...args) =>
  import('node-fetch').then(({ default: fetch }) => fetch(...args));

const app = express();

const IMG_SHAPE = 224;

app.use(express.json());
app.use(cors());

app.get('/', (req, res) => {
  res.send('Server working!')
})

app.post('/prediction', async (req, res) => {
  const { imageUrl } = req.body;
  const model = loadModel();

  if(imageUrl) {
    const response = await fetch(imageUrl);
    const imageBuffer = response && response.ok ? await response.buffer() : null;
    const imageTensor = tf.tidy(() => {
      const decode = tfn.node.decodeImage(imageBuffer, 3);
      const expand = tf.expandDims(decode, 0);
      return expand;
    });
    let imageTensorReshaped = tf.image.resizeBilinear(imageTensor,[IMG_SHAPE,IMG_SHAPE])
    console.log(imageTensorReshaped)
    await model.then((mod) => {
      mod.predict(imageTensorReshaped).data()
      .then(resp => {
        //console.log(indexOfMax(resp), resp[indexOfMax(resp)])
        const data = {
          index: indexOfMax(resp),
          prob: resp[indexOfMax(resp)]
        }
        console.log(data)
        res.json(data);
      }) 
    })
  }
})

const loadModel = async () => {
  const MODEL_URL = 'file://predict_food_tfjs/model.json'
  try {
    const food101Model = await tf.loadGraphModel(MODEL_URL);
    console.log('model loaded successfuly')
    return food101Model;
  } catch(error) {
    console.log(error)
  }
}

function indexOfMax(arr) {
  if (arr.length === 0) {
      return -1;
  }

  var max = arr[0];
  var maxIndex = 0;

  for (var i = 1; i < arr.length; i++) {
      if (arr[i] > max) {
          maxIndex = i;
          max = arr[i];
      }
  }

  return maxIndex;
}


app.listen(process.env.PORT || 3001, () => {
  console.log(`app is running on port ${process.env.PORT || 3001}`);
})
