import React, {useState} from 'react';
import {View, ImageBackground, StyleSheet, TextInput} from 'react-native';
import {Icon, Tooltip, Text} from 'react-native-elements'
import Spinner from 'react-native-loading-spinner-overlay';
import fetchImage from "../util/util";
import * as MediaLibrary from 'expo-media-library';
import * as FileSystem from "expo-file-system";
import IconButton from "../buttons/icon_button";


/**
 * Demo screen for image processing
 * @param props {navigation, route.params.imageUri}
 * @returns {JSX.Element}
 * @constructor
 */
export default function Demo({route, navigation}) {
    const {imageUri} = route.params;
    const [factor, setFactor] = useState('');
    const [sigma, setSigma] = useState('');
    const [size, setSize] = useState('');
    const [img, setImg] = useState(imageUri);
    const [spin, setSpin] = useState(false);
    const [clip, setClip] = useState('');
    const [blob, setBlob] = useState(null);

    const cartoonize = 'cartoon';
    const style_transfer = 'style_transfer';
    const histogram_vis = 'histogram_vis';
    const histogram = 'histogram';

    function setVal(val, setter, msg) {
        if (val === '') {
            setter('')
        } else {
            setter(msg + val)
        }
    }

    /**
     * Download current image showed on screen to album.
     * reference: https://stackoverflow.com/questions/60444307/save-blob-to-filesystem-in-react-native-expo
     */
    async function download() {
        if (img === imageUri) {
            await MediaLibrary.saveToLibraryAsync(imageUri).then(() => alert('saved successfully'))
        } else {
            const reader = new FileReader();
            reader.onload = async () => {
                const fileUri = `${FileSystem.cacheDirectory}/image.png`;
                const base64data = reader.result.split("data:image/png;base64,");

                await FileSystem.writeAsStringAsync(fileUri, base64data[1], {encoding: FileSystem.EncodingType.Base64});
                await MediaLibrary.saveToLibraryAsync(fileUri).then(() => alert('saved successfully'))
            }
            reader.readAsDataURL(blob);
        }
    }


    return (
        <View style={styles.container}>
            <Spinner
                visible={spin}
                textContent={'Loading...'}
            />
            <ImageBackground style={styles.image} source={{uri: img}}/>
            <View style={styles.buttonContainer}>
                <View style={styles.rowButtonContainer}>
                    <Tooltip height={100} popover={<Text>Scale the image based on input factor</Text>}>
                        <Icon name='ios-help-circle' type='ionicon' size={20} style={{padding: 5}}/>
                    </Tooltip>
                    <TextInput
                        placeholder="Scale factor"
                        onChangeText={val => setVal(val, setFactor, 'scale?factor=')}
                        defaultValue={''}
                        style={styles.input}
                    />
                    <Icon
                        name='ios-expand' type='ionicon' color='#517fa4'
                        style={{paddingLeft: 12, paddingTop: 3.5}}
                        size={30}
                        onPress={() => fetchImage(factor, imageUri, setSpin, setImg, setBlob)}
                    />
                </View>
                <View style={styles.rowButtonContainer}>
                    <Tooltip height={100} popover={<Text>Blur the image based on input sigma</Text>}>
                        <Icon name='ios-help-circle' type='ionicon' size={20} style={{padding: 5}}/>
                    </Tooltip>
                    <TextInput
                        placeholder="blur sigma"
                        onChangeText={val => setVal(val, setSigma, 'blur?sigma=')}
                        defaultValue={''}
                        style={styles.input}
                    />
                    <Icon name='blur-on' type='material' style={{paddingLeft: 5, paddingTop: 3.5}}
                          size={30}
                          onPress={() => fetchImage(sigma, imageUri, setSpin, setImg, setBlob)}
                    />
                </View>
                <View style={styles.rowButtonContainer}>
                    <Tooltip height={100}
                             popover={<Text>Resize the image based on input size by using seam carving</Text>}>
                        <Icon name='ios-help-circle' type='ionicon' size={20} style={{padding: 5}}/>
                    </Tooltip>
                    <TextInput
                        placeholder="Reduced size(e.g. put 15, 30x20 -> 15x20)"
                        onChangeText={val => setVal(val, setSize, 'seam_carving?reduced_size=')}
                        defaultValue={''}
                        style={styles.input}
                    />
                    <Icon name='photo-size-select-large' type='material' style={{paddingLeft: 5, paddingTop: 3.5}}
                          size={30}
                          onPress={() => fetchImage(size, imageUri, setSpin, setImg, setBlob)}
                    />
                </View>
                <View style={styles.rowButtonContainer}>
                    <Tooltip height={60} popover={<Text>CLAHE contrast enhancement</Text>}>
                        <Icon name='ios-help-circle' type='ionicon' size={20} style={{padding: 5}}/>
                    </Tooltip>
                    <TextInput
                        placeholder="clip threshold"
                        onChangeText={val => setVal(val, setClip, 'contrast?clip=')}
                        defaultValue={''}
                        style={styles.input}
                    />
                    <Icon name='adjust' type='font-awesome' style={{paddingLeft: 10, paddingTop: 3.5}}
                          size={30}
                          onPress={() => fetchImage(clip, imageUri, setSpin, setImg, setBlob)}
                    />
                </View>
                <View style={styles.bottomRowButtonContainer}>
                    <IconButton name={'ios-color-palette'} type={'ionicon'}
                                onPress={() => fetchImage(cartoonize, imageUri, setSpin, setImg, setBlob)}
                    />
                    <IconButton name='ios-color-wand' type='ionicon'
                                onPress={() => fetchImage(style_transfer, imageUri, setSpin, setImg, setBlob)}
                    />
                    <IconButton name='ios-stats' type='ionicon'
                                onPress={() => fetchImage(histogram_vis, imageUri, setSpin, setImg, setBlob)}
                    />
                    <IconButton name='ios-contrast' type='ionicon'
                                onPress={() => fetchImage(histogram, imageUri, setSpin, setImg, setBlob)}
                    />
                    <IconButton name='ios-swap' type='ionicon'
                                onPress={() => setImg(imageUri)}
                    />
                    <IconButton name='md-arrow-down' type='ionicon'
                                onPress={async () => await download()}
                    />
                </View>
                <View style={{position: "absolute", bottom: 10, left: 20}}>
                    <Tooltip height={200} popover={
                        <Text>Bottom row buttons, from left to right:
                            cartoonize, style transfer, histogram of pixel values, contrast enhancement
                            with histogram equalization, revert, save to album
                        </Text>
                    }>
                        <Icon name='ios-information-circle' type='ionicon' size={30}/>
                    </Tooltip>
                </View>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    buttonContainer: {
        flex: 1,
        flexDirection: "column",
    },
    image: {
        flex: 1,
        resizeMode: "cover",
        flexDirection: "column",
    },
    rowButtonContainer: {
        flexDirection: 'row',
        justifyContent: 'space-evenly',
        paddingTop: 10,
        paddingLeft: 10,
        paddingRight: 30,
        paddingBottom: 5,
        flexWrap: 'wrap',
    },
    bottomRowButtonContainer: {
        flexDirection: 'row',
        justifyContent: 'space-evenly',
        paddingTop: 10,
        paddingLeft: 10,
        paddingRight: 30,
        paddingBottom: 5,
        flexWrap: 'wrap',
        top: 40,
    },
    loading: {
        position: 'absolute',
        left: 0,
        right: 0,
        top: 0,
        bottom: 0,
        alignItems: 'center',
        justifyContent: 'center'
    },
    input: {
        flex: 1,
        paddingTop: 10,
        paddingRight: 10,
        paddingBottom: 10,
        paddingLeft: 0,
        backgroundColor: '#fff',
        color: '#424242',
    },
});
