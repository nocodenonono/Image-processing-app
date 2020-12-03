import React from 'react';
import {ImageBackground, StyleSheet, Text, View} from 'react-native';
import ImagePickerButton from "../buttons/pick_image_button";
import * as ImagePicker from "expo-image-picker";

/**
 * Home screen of the app
 * @param props navigation, navigate to other screens
 */
export default function Home(props) {
    /**
     * asks user permission to camera roll first and the let the user pick an image
     * finally navigates to the demo screen.
     * @returns {Promise<void>}
     */
    const pickImage = async () => {
        const {status} = await ImagePicker.requestCameraRollPermissionsAsync();

        if (status !== 'granted') {
            alert('Open up your settings to give me permissions :D');
        } else {
            const result = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ImagePicker.MediaTypeOptions.All,
                allowsEditing: true,
                aspect: [4, 3],
                quality: 1,
            });

            if (!result.cancelled) {
                props.navigation.navigate('image', {
                    imageUri: result.uri
                });
            }
        }
    };
    return (
        <View style={styles.container}>
            <ImageBackground source={require('../assets/back.jpeg')} style={styles.image}>
                <Text style={styles.text}>
                    Journey To Art
                </Text>
                <ImagePickerButton onPress={pickImage}/>
            </ImageBackground>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        flexDirection: "column"
    },
    image: {
        flex: 1,
        resizeMode: "cover",
        flexDirection: "column",
    },
    text: {
        fontSize: 42,
        fontWeight: "bold",
        textAlign: "center",
        color: "white",
        top:'33%'
    },
});
