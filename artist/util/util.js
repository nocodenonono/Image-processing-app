/**
 * fetch image from the image processing api
 * @param text query string
 * @param imageUri image uri
 * @param setSpin setSpin from Demo screen
 * @param setImg setImg from Demo screen
 * @param setBlob set blob object
 * @returns {Promise<void>}
 */
const fetchImage = async (text, imageUri, setSpin, setImg, setBlob) => {
    if (text === '') {
        alert('input cannot be empty');
        return;
    }
    setTimeout(() => {
        setSpin(true)
    }, 10);

    const data = new FormData()
    data.append('image', {
        uri: imageUri, // your file path string
        name: 'image.jpg',
    })

    return await fetch('http://127.0.0.1:5000/' + text, {
        method: 'POST',
        headers: {
            'Content-Type': 'multipart/form-data',
        },
        body: data
    }).then(response => {
        if (response.ok) {
            return response.blob()
        }
    }).then(function (myBlob) {
        const objectURL = URL.createObjectURL(myBlob);
        setImg(objectURL);
        setSpin(false);
        setBlob(myBlob)
    }).catch(() => {
        setSpin(false);
        alert('Please reselect your image or check your input or check your Internet');
    })
}

export default fetchImage