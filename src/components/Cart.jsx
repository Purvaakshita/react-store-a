import React from 'react'
import { appContext } from '../App'
import { useContext } from 'react'
export default function Cart() {
  const {Cart,products}=useContext(appContext)
  return (
    <div>
      {products.map(value=>(
        <div>{value.name}</div>
      ))}
    </div>
  )

}