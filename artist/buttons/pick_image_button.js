import React from 'react';
import { StyleSheet, Text, TouchableOpacity} from 'react-native';

/**
 * A button that lets user pick an image
 * @param props
 * @returns {*}
 * @constructor
 */
export default function ImagePickerButton(props) {

    return (
        <TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={props.onPress}>
                <Text style={styles.buttonText}>Start</Text>
            </TouchableOpacity>
        </TouchableOpacity>
    )
}

const styles = StyleSheet.create({
    buttonText: {
        fontSize: 25,
        fontWeight: 'bold',
        fontStyle: "italic",
    },
    button: {
        alignItems: "center",
        top: 630,
        backgroundColor:'#68a0cf',
        borderRadius: 100,
        marginLeft: 100,
        marginRight: 100,
        borderWidth: 1,
    },
});

