import fetchMock from 'jest-fetch-mock';
import React from "react";
import fetchImage from "../util/util";
import 'cross-fetch/polyfill';


test('tests network failure', () => {
    window.alert = jest.fn()
    fetchMock.once(() => {
        Promise.reject('Network failure')
    })
    fetchImage('test', 'test', jest.fn(), jest.fn(), jest.fn())
        .catch(error => {
            expect(error).toEqual('Network failure')
        })
})